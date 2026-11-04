// swiftrel_fast.h -- SwiftRel fast path for vcftools --relatedness2. Part of SwiftRel (LGPL-3.0).
#ifndef SWIFTREL_FAST_H
#define SWIFTREL_FAST_H

#include <string>
#include <vector>

#include "parameters.h"
#include "swiftrel_input.h"

namespace swiftrel {

// True if every active site filter is one the fast path reproduces exactly.
bool fast_path_supported(const parameters &params);

// Returns 0 on success (output written to out_fd), 1 if the input needs the original vcftools loop
// (nothing was written and `in` was not advanced), 2 on a write error. Fatal vcftools errors exit via LOG.error.
int run_fast_relatedness(const parameters &params, Input *in, unsigned N_indv, const std::vector<bool> &include_indv,
                         const std::vector<std::string> &indv_names, int out_fd, unsigned &N_entries,
                         unsigned &N_kept_entries);

}  // namespace swiftrel

#endif
