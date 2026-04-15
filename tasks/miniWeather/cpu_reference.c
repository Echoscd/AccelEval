// cpu_reference.c — miniWeather CPU baseline (compute_only interface)
//
// Adapted from miniWeather_standalone.cpp by Matt Norman (ORNL).
// Simulates dry, stratified, compressible, non-hydrostatic fluid flows.
// All compile-time constants converted to runtime parameters.
//
// Build: gcc -O2 -DORBENCH_COMPUTE_ONLY -I framework/
//        framework/harness_cpu.c tasks/miniWeather/task_io_cpu.c
//        tasks/miniWeather/cpu_reference.c -o solution_cpu -lm

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// ===== Physical constants =====
static const double pi        = 3.14159265358979323846264338327;
static const double grav      = 9.8;
static const double cp        = 1004.;
static const double cv        = 717.;
static const double rd        = 287.;
static const double p0        = 1.e5;
static const double C0        = 27.5629410929725921310572974482;
static const double gamm      = 1.40027894002789400278940027894;
static const double xlen      = 2.e4;
static const double zlen      = 1.e4;
static const double hv_beta   = 0.05;
static const double cfl       = 1.50;
static const double max_speed = 450;
static const int    hs        = 2;
static const int    sten_size = 4;
static const int    NUM_VARS  = 4;
static const int    ID_DENS   = 0;
static const int    ID_UMOM   = 1;
static const int    ID_WMOM   = 2;
static const int    ID_RHOT   = 3;
static const int    DIR_X     = 1;
static const int    DIR_Z     = 2;
static const int    DATA_SPEC_COLLISION       = 1;
static const int    DATA_SPEC_THERMAL         = 2;
static const int    DATA_SPEC_GRAVITY_WAVES   = 3;
static const int    DATA_SPEC_DENSITY_CURRENT = 5;
static const int    DATA_SPEC_INJECTION       = 6;
static const int    nqpoints  = 3;
static const double qpoints[] = {0.112701665379258311482073460022, 0.5, 0.887298334620741688517926539980};
static const double qweights[]= {0.277777777777777777777777777779, 0.444444444444444444444444444444, 0.277777777777777777777777777779};

// ===== Runtime parameters (set in solution_compute) =====
static int    g_nx, g_nz;
static int    g_data_spec;
static double g_sim_time;
static double g_dx, g_dz, g_dt;

// ===== Simulation state =====
static double *g_state, *g_state_tmp, *g_flux, *g_tend;
static double *g_hy_dens_cell, *g_hy_dens_theta_cell;
static double *g_hy_dens_int, *g_hy_dens_theta_int, *g_hy_pressure_int;

static double dmin(double a, double b) { return a < b ? a : b; }

// ===== Initial condition functions =====

static void hydro_const_theta(double z, double *r, double *t) {
    double theta0 = 300., exner0 = 1.;
    *t = theta0;
    double exner = exner0 - grav * z / (cp * theta0);
    double p = p0 * pow(exner, cp / rd);
    double rt = pow(p / C0, 1. / gamm);
    *r = rt / *t;
}

static void hydro_const_bvfreq(double z, double bv_freq0, double *r, double *t) {
    double theta0 = 300., exner0 = 1.;
    *t = theta0 * exp(bv_freq0 * bv_freq0 / grav * z);
    double exner = exner0 - grav * grav / (cp * bv_freq0 * bv_freq0) * (*t - theta0) / (*t * theta0);
    double p = p0 * pow(exner, cp / rd);
    double rt = pow(p / C0, 1. / gamm);
    *r = rt / *t;
}

static double sample_ellipse_cosine(double x, double z, double amp, double x0, double z0, double xrad, double zrad) {
    double dist = sqrt(((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad)) * pi / 2.;
    if (dist <= pi / 2.) return amp * pow(cos(dist), 2.);
    return 0.;
}

static void thermal(double x, double z, double *r, double *u, double *w, double *t, double *hr, double *ht) {
    hydro_const_theta(z, hr, ht);
    *r = 0.; *t = 0.; *u = 0.; *w = 0.;
    *t += sample_ellipse_cosine(x, z, 3., xlen/2, 2000., 2000., 2000.);
}

static void collision(double x, double z, double *r, double *u, double *w, double *t, double *hr, double *ht) {
    hydro_const_theta(z, hr, ht);
    *r = 0.; *t = 0.; *u = 0.; *w = 0.;
    *t += sample_ellipse_cosine(x, z, 20., xlen/2, 2000., 2000., 2000.);
    *t += sample_ellipse_cosine(x, z, -20., xlen/2, 8000., 2000., 2000.);
}

static void density_current(double x, double z, double *r, double *u, double *w, double *t, double *hr, double *ht) {
    hydro_const_theta(z, hr, ht);
    *r = 0.; *t = 0.; *u = 0.; *w = 0.;
    *t += sample_ellipse_cosine(x, z, -20., xlen/2, 5000., 4000., 2000.);
}

static void gravity_waves(double x, double z, double *r, double *u, double *w, double *t, double *hr, double *ht) {
    hydro_const_bvfreq(z, 0.02, hr, ht);
    *r = 0.; *t = 0.; *u = 15.; *w = 0.;
}

static void injection(double x, double z, double *r, double *u, double *w, double *t, double *hr, double *ht) {
    hydro_const_theta(z, hr, ht);
    *r = 0.; *t = 0.; *u = 0.; *w = 0.;
}

static void sample_initial_condition(double x, double z, int spec,
                                     double *r, double *u, double *w, double *t, double *hr, double *ht) {
    if      (spec == DATA_SPEC_COLLISION)       collision(x, z, r, u, w, t, hr, ht);
    else if (spec == DATA_SPEC_THERMAL)         thermal(x, z, r, u, w, t, hr, ht);
    else if (spec == DATA_SPEC_GRAVITY_WAVES)   gravity_waves(x, z, r, u, w, t, hr, ht);
    else if (spec == DATA_SPEC_DENSITY_CURRENT) density_current(x, z, r, u, w, t, hr, ht);
    else if (spec == DATA_SPEC_INJECTION)       injection(x, z, r, u, w, t, hr, ht);
    else                                        thermal(x, z, r, u, w, t, hr, ht);
}

// ===== Simulation functions =====

static void sim_init(void) {
    int nx = g_nx, nz = g_nz;
    double dx = g_dx, dz = g_dz;
    int i, k, ii, kk, ll, inds;
    double x, z, r, u, w, t, hr, ht;

    g_state     = (double*)malloc((nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double));
    g_state_tmp = (double*)malloc((nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double));
    g_flux      = (double*)malloc((nx+1)*(nz+1)*NUM_VARS*sizeof(double));
    g_tend      = (double*)malloc(nx*nz*NUM_VARS*sizeof(double));
    g_hy_dens_cell       = (double*)malloc((nz+2*hs)*sizeof(double));
    g_hy_dens_theta_cell = (double*)malloc((nz+2*hs)*sizeof(double));
    g_hy_dens_int        = (double*)malloc((nz+1)*sizeof(double));
    g_hy_dens_theta_int  = (double*)malloc((nz+1)*sizeof(double));
    g_hy_pressure_int    = (double*)malloc((nz+1)*sizeof(double));

    g_dt = dmin(dx, dz) / max_speed * cfl;

    // Initialize state via Gauss-Legendre quadrature
    for (k = 0; k < nz+2*hs; k++) {
        for (i = 0; i < nx+2*hs; i++) {
            for (ll = 0; ll < NUM_VARS; ll++) {
                inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
                g_state[inds] = 0.;
            }
            for (kk = 0; kk < nqpoints; kk++) {
                for (ii = 0; ii < nqpoints; ii++) {
                    x = (i - hs + 0.5)*dx + (qpoints[ii]-0.5)*dx;
                    z = (k - hs + 0.5)*dz + (qpoints[kk]-0.5)*dz;
                    sample_initial_condition(x, z, g_data_spec, &r, &u, &w, &t, &hr, &ht);
                    inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
                    g_state[inds] += r * qweights[ii]*qweights[kk];
                    inds = ID_UMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
                    g_state[inds] += (r+hr)*u * qweights[ii]*qweights[kk];
                    inds = ID_WMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
                    g_state[inds] += (r+hr)*w * qweights[ii]*qweights[kk];
                    inds = ID_RHOT*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
                    g_state[inds] += ((r+hr)*(t+ht) - hr*ht) * qweights[ii]*qweights[kk];
                }
            }
            for (ll = 0; ll < NUM_VARS; ll++) {
                inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
                g_state_tmp[inds] = g_state[inds];
            }
        }
    }

    // Hydrostatic background state (cell averages)
    for (k = 0; k < nz+2*hs; k++) {
        double r_tmp, u_tmp, w_tmp, t_tmp, hr_tmp, ht_tmp;
        g_hy_dens_cell[k] = 0.;
        g_hy_dens_theta_cell[k] = 0.;
        for (kk = 0; kk < nqpoints; kk++) {
            z = (k - hs + 0.5)*dz;
            sample_initial_condition(0., z, g_data_spec, &r_tmp, &u_tmp, &w_tmp, &t_tmp, &hr_tmp, &ht_tmp);
            g_hy_dens_cell[k]       += hr_tmp * qweights[kk];
            g_hy_dens_theta_cell[k] += hr_tmp * ht_tmp * qweights[kk];
        }
    }

    // Hydrostatic background state (cell interfaces)
    for (k = 0; k < nz+1; k++) {
        double r_tmp, u_tmp, w_tmp, t_tmp, hr_tmp, ht_tmp;
        z = k * dz;
        sample_initial_condition(0., z, g_data_spec, &r_tmp, &u_tmp, &w_tmp, &t_tmp, &hr_tmp, &ht_tmp);
        g_hy_dens_int[k]       = hr_tmp;
        g_hy_dens_theta_int[k] = hr_tmp * ht_tmp;
        g_hy_pressure_int[k]   = C0 * pow(hr_tmp * ht_tmp, gamm);
    }
}

static void sim_cleanup(void) {
    free(g_state);       g_state = NULL;
    free(g_state_tmp);   g_state_tmp = NULL;
    free(g_flux);        g_flux = NULL;
    free(g_tend);        g_tend = NULL;
    free(g_hy_dens_cell);       g_hy_dens_cell = NULL;
    free(g_hy_dens_theta_cell); g_hy_dens_theta_cell = NULL;
    free(g_hy_dens_int);        g_hy_dens_int = NULL;
    free(g_hy_dens_theta_int);  g_hy_dens_theta_int = NULL;
    free(g_hy_pressure_int);    g_hy_pressure_int = NULL;
}

static void set_halo_values_x(double *state) {
    int nx = g_nx, nz = g_nz;
    double dz_l = g_dz;
    int k, ll, i;
    for (ll = 0; ll < NUM_VARS; ll++) {
        for (k = 0; k < nz; k++) {
            state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + 0]       = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs-2];
            state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + 1]       = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs-1];
            state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs]   = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + hs];
            state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs+1] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + hs+1];
        }
    }
    if (g_data_spec == DATA_SPEC_INJECTION) {
        for (k = 0; k < nz; k++) {
            for (i = 0; i < hs; i++) {
                double z = (k + 0.5) * dz_l;
                if (fabs(z - 3*zlen/4) <= zlen/16) {
                    int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
                    int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
                    int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
                    state[ind_u] = (state[ind_r] + g_hy_dens_cell[k+hs]) * 50.;
                    state[ind_t] = (state[ind_r] + g_hy_dens_cell[k+hs]) * 298. - g_hy_dens_theta_cell[k+hs];
                }
            }
        }
    }
}

static void set_halo_values_z(double *state) {
    int nx = g_nx, nz = g_nz;
    int i, ll;
    for (ll = 0; ll < NUM_VARS; ll++) {
        for (i = 0; i < nx+2*hs; i++) {
            if (ll == ID_WMOM) {
                state[ll*(nz+2*hs)*(nx+2*hs) + 0*(nx+2*hs) + i]         = 0.;
                state[ll*(nz+2*hs)*(nx+2*hs) + 1*(nx+2*hs) + i]         = 0.;
                state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs)*(nx+2*hs) + i]   = 0.;
                state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = 0.;
            } else if (ll == ID_UMOM) {
                state[ll*(nz+2*hs)*(nx+2*hs) + 0*(nx+2*hs) + i]         = state[ll*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i] / g_hy_dens_cell[hs] * g_hy_dens_cell[0];
                state[ll*(nz+2*hs)*(nx+2*hs) + 1*(nx+2*hs) + i]         = state[ll*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i] / g_hy_dens_cell[hs] * g_hy_dens_cell[1];
                state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs)*(nx+2*hs) + i]   = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i] / g_hy_dens_cell[nz+hs-1] * g_hy_dens_cell[nz+hs];
                state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i] / g_hy_dens_cell[nz+hs-1] * g_hy_dens_cell[nz+hs+1];
            } else {
                state[ll*(nz+2*hs)*(nx+2*hs) + 0*(nx+2*hs) + i]         = state[ll*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i];
                state[ll*(nz+2*hs)*(nx+2*hs) + 1*(nx+2*hs) + i]         = state[ll*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i];
                state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs)*(nx+2*hs) + i]   = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
                state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
            }
        }
    }
}

static void compute_tendencies_x(double *state, double *flux, double *tend, double dt) {
    int nx = g_nx, nz = g_nz;
    double dx = g_dx;
    int i, k, ll, s, inds, indf1, indf2, indt;
    double r, u, w, t, p, stencil[4], d3_vals[4], vals[4], hv_coef;
    hv_coef = -hv_beta * dx / (16 * dt);
    for (k = 0; k < nz; k++) {
        for (i = 0; i < nx+1; i++) {
            for (ll = 0; ll < NUM_VARS; ll++) {
                for (s = 0; s < sten_size; s++) {
                    inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+s;
                    stencil[s] = state[inds];
                }
                vals[ll]    = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
                d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
            }
            r = vals[ID_DENS] + g_hy_dens_cell[k+hs];
            u = vals[ID_UMOM] / r;
            w = vals[ID_WMOM] / r;
            t = (vals[ID_RHOT] + g_hy_dens_theta_cell[k+hs]) / r;
            p = C0 * pow(r*t, gamm);
            flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u     - hv_coef*d3_vals[ID_DENS];
            flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
            flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*w   - hv_coef*d3_vals[ID_WMOM];
            flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*t   - hv_coef*d3_vals[ID_RHOT];
        }
    }
    for (ll = 0; ll < NUM_VARS; ll++) {
        for (k = 0; k < nz; k++) {
            for (i = 0; i < nx; i++) {
                indt  = ll*nz*nx + k*nx + i;
                indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i;
                indf2 = ll*(nz+1)*(nx+1) + k*(nx+1) + i+1;
                tend[indt] = -(flux[indf2] - flux[indf1]) / dx;
            }
        }
    }
}

static void compute_tendencies_z(double *state, double *flux, double *tend, double dt) {
    int nx = g_nx, nz = g_nz;
    double dz = g_dz;
    int i, k, ll, s, inds, indf1, indf2, indt;
    double r, u, w, t, p, stencil[4], d3_vals[4], vals[4], hv_coef;
    hv_coef = -hv_beta * dz / (16 * dt);
    for (k = 0; k < nz+1; k++) {
        for (i = 0; i < nx; i++) {
            for (ll = 0; ll < NUM_VARS; ll++) {
                for (s = 0; s < sten_size; s++) {
                    inds = ll*(nz+2*hs)*(nx+2*hs) + (k+s)*(nx+2*hs) + i+hs;
                    stencil[s] = state[inds];
                }
                vals[ll]    = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
                d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
            }
            r = vals[ID_DENS] + g_hy_dens_int[k];
            u = vals[ID_UMOM] / r;
            w = vals[ID_WMOM] / r;
            t = (vals[ID_RHOT] + g_hy_dens_theta_int[k]) / r;
            p = C0 * pow(r*t, gamm) - g_hy_pressure_int[k];
            if (k == 0 || k == nz) { w = 0; d3_vals[ID_DENS] = 0; }
            flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w     - hv_coef*d3_vals[ID_DENS];
            flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
            flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
            flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
        }
    }
    for (ll = 0; ll < NUM_VARS; ll++) {
        for (k = 0; k < nz; k++) {
            for (i = 0; i < nx; i++) {
                indt  = ll*nz*nx + k*nx + i;
                indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i;
                indf2 = ll*(nz+1)*(nx+1) + (k+1)*(nx+1) + i;
                tend[indt] = -(flux[indf2] - flux[indf1]) / dz;
                if (ll == ID_WMOM) {
                    inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
                    tend[indt] -= state[inds] * grav;
                }
            }
        }
    }
}

static void semi_discrete_step(double *state_init, double *state_forcing, double *state_out,
                                double dt, int dir, double *flux, double *tend) {
    int nx = g_nx, nz = g_nz;
    int i, k, ll, inds, indt;
    if (dir == DIR_X) {
        set_halo_values_x(state_forcing);
        compute_tendencies_x(state_forcing, flux, tend, dt);
    } else {
        set_halo_values_z(state_forcing);
        compute_tendencies_z(state_forcing, flux, tend, dt);
    }
    for (ll = 0; ll < NUM_VARS; ll++) {
        for (k = 0; k < nz; k++) {
            for (i = 0; i < nx; i++) {
                inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
                indt = ll*nz*nx + k*nx + i;
                state_out[inds] = state_init[inds] + dt * tend[indt];
            }
        }
    }
}

static void perform_timestep(double *state, double *state_tmp, double *flux, double *tend,
                              double dt, int *direction_switch) {
    if (*direction_switch) {
        semi_discrete_step(state, state,     state_tmp, dt/3, DIR_X, flux, tend);
        semi_discrete_step(state, state_tmp, state_tmp, dt/2, DIR_X, flux, tend);
        semi_discrete_step(state, state_tmp, state,     dt/1, DIR_X, flux, tend);
        semi_discrete_step(state, state,     state_tmp, dt/3, DIR_Z, flux, tend);
        semi_discrete_step(state, state_tmp, state_tmp, dt/2, DIR_Z, flux, tend);
        semi_discrete_step(state, state_tmp, state,     dt/1, DIR_Z, flux, tend);
    } else {
        semi_discrete_step(state, state,     state_tmp, dt/3, DIR_Z, flux, tend);
        semi_discrete_step(state, state_tmp, state_tmp, dt/2, DIR_Z, flux, tend);
        semi_discrete_step(state, state_tmp, state,     dt/1, DIR_Z, flux, tend);
        semi_discrete_step(state, state,     state_tmp, dt/3, DIR_X, flux, tend);
        semi_discrete_step(state, state_tmp, state_tmp, dt/2, DIR_X, flux, tend);
        semi_discrete_step(state, state_tmp, state,     dt/1, DIR_X, flux, tend);
    }
    *direction_switch = !(*direction_switch);
}

static void reductions(double *state, double *mass_out, double *te_out) {
    int nx = g_nx, nz = g_nz;
    double dx = g_dx, dz = g_dz;
    double mass = 0, te = 0;
    int k, i;
    for (k = 0; k < nz; k++) {
        for (i = 0; i < nx; i++) {
            int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
            int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
            int ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
            int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
            double r  = state[ind_r] + g_hy_dens_cell[hs+k];
            double u  = state[ind_u] / r;
            double w  = state[ind_w] / r;
            double th = (state[ind_t] + g_hy_dens_theta_cell[hs+k]) / r;
            double p  = C0 * pow(r*th, gamm);
            double tt = th / pow(p0/p, rd/cp);
            double ke = r * (u*u + w*w);
            double ie = r * cv * tt;
            mass += r * dx * dz;
            te   += (ke + ie) * dx * dz;
        }
    }
    *mass_out = mass;
    *te_out   = te;
}

// ===== ORBench compute_only interface =====

void solution_compute(
    int nx_in, int nz_in,
    int sim_time_in, int data_spec_in,
    double *output  // [6]: d_mass, d_te, L2_dens, L2_umom, L2_wmom, L2_rhot
) {
    // Set runtime parameters
    g_nx = nx_in;
    g_nz = nz_in;
    g_sim_time  = (double)sim_time_in;
    g_data_spec = data_spec_in;
    g_dx = xlen / g_nx;
    g_dz = zlen / g_nz;

    // Initialize simulation
    sim_init();

    // Initial reductions
    double mass0, te0;
    reductions(g_state, &mass0, &te0);

    // Main time loop
    double etime = 0.;
    double dt = g_dt;
    int direction_switch = 1;
    while (etime < g_sim_time) {
        if (etime + dt > g_sim_time) dt = g_sim_time - etime;
        perform_timestep(g_state, g_state_tmp, g_flux, g_tend, dt, &direction_switch);
        etime += dt;
    }

    // Final reductions
    double mass, te;
    reductions(g_state, &mass, &te);

    output[0] = (mass - mass0) / mass0;  // d_mass
    output[1] = (te - te0) / te0;        // d_te

    // L2 norms of each state variable (perturbation fields)
    int nx = g_nx, nz = g_nz;
    double l2[4] = {0, 0, 0, 0};
    int k, i, ll;
    for (ll = 0; ll < NUM_VARS; ll++) {
        for (k = 0; k < nz; k++) {
            for (i = 0; i < nx; i++) {
                int ind = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
                l2[ll] += g_state[ind] * g_state[ind];
            }
        }
        l2[ll] = sqrt(l2[ll]);
    }
    output[2] = l2[ID_DENS];  // L2 density perturbation
    output[3] = l2[ID_UMOM];  // L2 x-momentum
    output[4] = l2[ID_WMOM];  // L2 z-momentum
    output[5] = l2[ID_RHOT];  // L2 rho*theta perturbation

    // Cleanup
    sim_cleanup();
}

void solution_free(void) {}
