// Metadata-only probe for this demo's pinned, one-slot, non-speculative server.
// Server parity: tools/server/server-context.cpp server_output_limits()/load_model().
// Keep upstream inference, kernels, model loading and memory accounting unchanged.
#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#ifndef PD_PROBE_SOURCE_SHA256
#error "The probe must embed its source provenance"
#endif
#ifndef PD_ENGINE_REVISION
#error "The probe must be built against the verified pinned revision"
#endif

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("pd-server-fit-params server-context-v1 engine=%s linked_commit=%s source_sha256=%s\n",
                    PD_ENGINE_REVISION, llama_commit(), PD_PROBE_SOURCE_SHA256);
        return 0;
    }

    // --fit-print is a fit-tool-only option. Accept only its no-allocation mode,
    // then reuse SERVER parsing so every remaining option/default matches it.
    std::vector<char *> arguments = { argv[0] };
    bool print_requested = false;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--fit-print") == 0) {
            if (index + 1 >= argc || std::strcmp(argv[++index], "on") != 0) {
                std::fprintf(stderr, "server probe requires --fit-print on\n");
                return 1;
            }
            print_requested = true;
        } else {
            arguments.push_back(argv[index]);
        }
    }
    if (!print_requested) {
        std::fprintf(stderr, "server probe requires --fit-print on\n");
        return 1;
    }

    arguments.push_back(nullptr); // Preserve argv[argc] == nullptr.
    common_init();
    common_params params;
    if (!common_params_parse(static_cast<int>(arguments.size() - 1), arguments.data(), params,
                             LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    const bool spec_none = std::all_of(params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type type) { return type == COMMON_SPECULATIVE_TYPE_NONE; });
    if (params.n_parallel != 1 || params.fit_params || params.embedding || params.sampling.backend_sampling ||
            params.speculative.has_dft() || params.speculative.has_synth() ||
            !spec_none || !params.mmproj.path.empty() ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
             params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        std::fprintf(stderr, "server probe supports only parallel=1, fit=off, text generation, spec=none\n");
        return 1;
    }

    // server_output_limits() returns {parallel, 1} for text/spec=none. Apply
    // these BEFORE the same conversion used by common_init_from_params().
    params.n_outputs_max = params.n_parallel;
    params.n_outputs_max_per_seq = 1;
    auto model = common_model_params_to_llama(params);
    auto context = common_context_params_to_llama(params);
    if (context.n_rs_seq != 0) {
        std::fprintf(stderr, "server probe requires n_rs_seq=0\n");
        return 1;
    }
    std::fprintf(stderr,
        "PD_SERVER_FIT_PARAMS {\"kind\":\"server-context-v1\",\"engine_revision\":\"%s\","
        "\"linked_commit\":\"%s\",\"probe_source_sha256\":\"%s\",\"parallel\":%u,"
        "\"n_outputs_max\":%u,\"n_outputs_max_per_seq\":%u,\"n_rs_seq\":%u,"
        "\"n_ctx\":%u,\"n_batch\":%u,\"n_ubatch\":%u,\"kv_unified\":%s,"
        "\"offload_kqv\":%s,\"op_offload\":%s,\"flash_attn_type\":%d}\n",
        PD_ENGINE_REVISION, llama_commit(), PD_PROBE_SOURCE_SHA256, context.n_seq_max,
        context.n_outputs_max, context.n_outputs_max_per_seq, context.n_rs_seq,
        context.n_ctx, context.n_batch, context.n_ubatch,
        context.kv_unified ? "true" : "false", context.offload_kqv ? "true" : "false",
        context.op_offload ? "true" : "false", static_cast<int>(context.flash_attn_type));

    llama_backend_init();
    llama_numa_init(params.numa);
    try {
        // Upstream common_fit_print forces model.no_alloc=true/load_mode=NONE.
        // It builds the exact reserved graph without reading/allocating weights.
        common_fit_print(params.model.path.c_str(), &model, &context);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "server probe failed: %s\n", error.what());
        llama_backend_free();
        return 1;
    }
    llama_backend_free();
    return 0;
}
