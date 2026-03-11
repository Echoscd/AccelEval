// orbench_io.h - ORBench v2 binary input + request I/O (C, shared by CPU/GPU harness)
//
// NOTE:
// - This header is included by both .c and .cu translation units.
// - Keep it strictly C (no C++ features).
// - Implements:
//   TaskData  load_input_bin(const char* bin_path);
//   void      load_requests(const char* txt_path, char** requests, int* count);
//   void      free_task_data(TaskData* data);
//   int64_t   get_param(const TaskData* data, const char* key);
//   void*     get_tensor(const TaskData* data, const char* name);
//   int*      get_tensor_int(const TaskData* data, const char* name);
//   float*    get_tensor_float(const TaskData* data, const char* name);

#ifndef ORBENCH_IO_H
#define ORBENCH_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===== input.bin format =====
// FileHeader (32 bytes)
typedef struct {
    char     magic[8];      // "ORBENCH\0"
    int32_t  version;       // 1
    int32_t  num_tensors;   // number of TensorDesc entries
    int32_t  num_params;    // number of ParamEntry entries
    int32_t  data_offset;   // start offset of raw data region (64B aligned)
    int64_t  reserved;      // 0
} FileHeader;

// TensorDesc (64 bytes)
typedef struct {
    char     name[32];      // tensor name
    int32_t  dtype;         // 0=INT32, 1=FLOAT32, 2=FLOAT64
    int32_t  reserved;
    int64_t  count;         // number of elements (1D)
    int64_t  offset;        // byte offset in file
    int64_t  size_bytes;    // bytes
} TensorDesc;

// ParamEntry (48 bytes)
typedef struct {
    char     key[32];
    int64_t  value;
} ParamEntry;

// ===== parsed data passed to solution =====
typedef struct {
    const char* name;
    int         dtype;      // 0=INT32, 1=FLOAT32, 2=FLOAT64 (matches TensorDesc dtype)
    void*       data;       // malloc'ed host pointer
    int64_t     count;
} Tensor;

typedef struct {
    const char* key;
    int64_t     value;
} OrbenchParamKV;

typedef struct {
    int      num_inputs;
    Tensor*  inputs;
    int      num_params;
    // key/value list (malloc'ed, owned by TaskData)
    OrbenchParamKV* params;
} TaskData;

// ===== helpers =====
static int64_t _orbench_align64(int64_t x) {
    return (x + 63) & ~((int64_t)63);
}

static int _orbench_check_magic(const char magic[8]) {
    const char expected[8] = { 'O','R','B','E','N','C','H','\0' };
    return memcmp(magic, expected, 8) == 0;
}

static int _orbench_dtype_size(int32_t dtype) {
    if (dtype == 0) return 4;     // int32
    if (dtype == 1) return 4;     // float32
    if (dtype == 2) return 8;     // float64
    return 0;
}

static void _orbench_fatal(const char* msg, const char* path) {
    if (path) fprintf(stderr, "[orbench_io] %s: %s\n", msg, path);
    else fprintf(stderr, "[orbench_io] %s\n", msg);
    exit(1);
}

// ===== API =====
static TaskData load_input_bin(const char* bin_path) {
    TaskData data;
    memset(&data, 0, sizeof(TaskData));

    FILE* f = fopen(bin_path, "rb");
    if (!f) _orbench_fatal("Cannot open input.bin", bin_path);

    FileHeader hdr;
    if (fread(&hdr, 1, sizeof(FileHeader), f) != sizeof(FileHeader)) {
        fclose(f);
        _orbench_fatal("Failed to read FileHeader", bin_path);
    }

    if (!_orbench_check_magic(hdr.magic)) {
        fclose(f);
        _orbench_fatal("Bad magic (expected ORBENCH\\0)", bin_path);
    }
    if (hdr.version != 1) {
        fclose(f);
        _orbench_fatal("Unsupported version", bin_path);
    }
    if (hdr.num_tensors < 0 || hdr.num_params < 0) {
        fclose(f);
        _orbench_fatal("Invalid counts in header", bin_path);
    }
    if (hdr.data_offset <= 0 || (hdr.data_offset % 64) != 0) {
        fclose(f);
        _orbench_fatal("data_offset must be positive and 64B aligned", bin_path);
    }

    // Read TensorDesc array
    TensorDesc* descs = NULL;
    if (hdr.num_tensors > 0) {
        descs = (TensorDesc*)malloc((size_t)hdr.num_tensors * sizeof(TensorDesc));
        if (!descs) _orbench_fatal("OOM allocating TensorDesc", NULL);
        if (fread(descs, sizeof(TensorDesc), (size_t)hdr.num_tensors, f) != (size_t)hdr.num_tensors) {
            fclose(f);
            free(descs);
            _orbench_fatal("Failed to read TensorDesc array", bin_path);
        }
    }

    // Read ParamEntry array
    ParamEntry* params = NULL;
    if (hdr.num_params > 0) {
        params = (ParamEntry*)malloc((size_t)hdr.num_params * sizeof(ParamEntry));
        if (!params) _orbench_fatal("OOM allocating ParamEntry", NULL);
        if (fread(params, sizeof(ParamEntry), (size_t)hdr.num_params, f) != (size_t)hdr.num_params) {
            fclose(f);
            free(descs);
            free(params);
            _orbench_fatal("Failed to read ParamEntry array", bin_path);
        }
    }

    // Construct TaskData tensors (host malloc + fread raw)
    data.num_inputs = hdr.num_tensors;
    if (hdr.num_tensors > 0) {
        data.inputs = (Tensor*)calloc((size_t)hdr.num_tensors, sizeof(Tensor));
        if (!data.inputs) _orbench_fatal("OOM allocating Tensor list", NULL);
    }

    for (int i = 0; i < hdr.num_tensors; i++) {
        TensorDesc* td = &descs[i];
        int elem_size = _orbench_dtype_size(td->dtype);
        if (elem_size == 0) {
            fclose(f);
            free(descs);
            free(params);
            _orbench_fatal("Unsupported dtype in TensorDesc", td->name);
        }
        if (td->size_bytes != td->count * (int64_t)elem_size) {
            fclose(f);
            free(descs);
            free(params);
            _orbench_fatal("TensorDesc size_bytes mismatch", td->name);
        }

        void* buf = malloc((size_t)td->size_bytes);
        if (!buf) _orbench_fatal("OOM allocating tensor buffer", td->name);

        if (fseek(f, (long)td->offset, SEEK_SET) != 0) {
            fclose(f);
            free(buf);
            free(descs);
            free(params);
            _orbench_fatal("Failed to seek to tensor offset", td->name);
        }
        if (fread(buf, 1, (size_t)td->size_bytes, f) != (size_t)td->size_bytes) {
            fclose(f);
            free(buf);
            free(descs);
            free(params);
            _orbench_fatal("Failed to read tensor raw data", td->name);
        }

        // Copy name into a malloc'ed stable string (so td can be freed)
        size_t nlen = strnlen(td->name, 32);
        char* name_copy = (char*)malloc(nlen + 1);
        if (!name_copy) _orbench_fatal("OOM allocating tensor name", NULL);
        memcpy(name_copy, td->name, nlen);
        name_copy[nlen] = '\0';

        data.inputs[i].name = name_copy;
        data.inputs[i].dtype = td->dtype;
        data.inputs[i].data = buf;
        data.inputs[i].count = td->count;
    }

    // Construct params list with stable key strings
    data.num_params = hdr.num_params;
    if (hdr.num_params > 0) {
        data.params = (OrbenchParamKV*)calloc((size_t)hdr.num_params, sizeof(*data.params));
        if (!data.params) _orbench_fatal("OOM allocating params list", NULL);
    }
    for (int i = 0; i < hdr.num_params; i++) {
        size_t klen = strnlen(params[i].key, 32);
        char* key_copy = (char*)malloc(klen + 1);
        if (!key_copy) _orbench_fatal("OOM allocating param key", NULL);
        memcpy(key_copy, params[i].key, klen);
        key_copy[klen] = '\0';

        data.params[i].key = key_copy;
        data.params[i].value = params[i].value;
    }

    free(descs);
    free(params);
    fclose(f);
    return data;
}

static void load_requests(const char* txt_path, char** requests, int* count) {
    *count = 0;
    FILE* f = fopen(txt_path, "r");
    if (!f) _orbench_fatal("Cannot open requests.txt", txt_path);

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        // trim newline
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
            line[n - 1] = '\0';
            n--;
        }
        if (n == 0) continue;
        char* s = (char*)malloc(n + 1);
        if (!s) _orbench_fatal("OOM allocating request string", NULL);
        memcpy(s, line, n + 1);
        requests[*count] = s;
        (*count)++;
    }
    fclose(f);
}

static void free_task_data(TaskData* data) {
    if (!data) return;
    for (int i = 0; i < data->num_inputs; i++) {
        if (data->inputs) {
            if (data->inputs[i].data) free(data->inputs[i].data);
            if (data->inputs[i].name) free((void*)data->inputs[i].name);
        }
    }
    if (data->inputs) free(data->inputs);

    for (int i = 0; i < data->num_params; i++) {
        if (data->params && data->params[i].key) free((void*)data->params[i].key);
    }
    if (data->params) free(data->params);

    memset(data, 0, sizeof(TaskData));
}

static int64_t get_param(const TaskData* data, const char* key) {
    if (!data || !key) return 0;
    for (int i = 0; i < data->num_params; i++) {
        if (strcmp(data->params[i].key, key) == 0) return data->params[i].value;
    }
    return 0;
}

static void* get_tensor(const TaskData* data, const char* name) {
    if (!data || !name) return NULL;
    for (int i = 0; i < data->num_inputs; i++) {
        if (strcmp(data->inputs[i].name, name) == 0) return data->inputs[i].data;
    }
    return NULL;
}

static int* get_tensor_int(const TaskData* data, const char* name) {
    // dtype 0
    if (!data || !name) return NULL;
    for (int i = 0; i < data->num_inputs; i++) {
        if (strcmp(data->inputs[i].name, name) == 0) {
            if (data->inputs[i].dtype != 0) return NULL;
            return (int*)data->inputs[i].data;
        }
    }
    return NULL;
}

static float* get_tensor_float(const TaskData* data, const char* name) {
    // dtype 1
    if (!data || !name) return NULL;
    for (int i = 0; i < data->num_inputs; i++) {
        if (strcmp(data->inputs[i].name, name) == 0) {
            if (data->inputs[i].dtype != 1) return NULL;
            return (float*)data->inputs[i].data;
        }
    }
    return NULL;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ORBENCH_IO_H


