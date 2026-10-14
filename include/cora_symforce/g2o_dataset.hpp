#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pose2.hpp"

namespace cora_symforce {

struct G2oDataset {
  std::vector<Pose2State> poses;
  std::vector<RelativePoseMeasurement> measurements;
  std::size_t vertex_records = 0;
  bool has_complete_vertex_initialization = false;
};

// SE-Sync's isotropic translational precision conversion for EDGE_SE2:
// tau = 2 / trace(TranInfo^{-1}), where
// TranInfo = [[I11, I12], [I12, I22]].
double seSyncTauFromTranslationInformation(
    double I11,
    double I12,
    double I22);

// Load a 2D g2o pose graph containing EDGE_SE2 measurements. VERTEX_SE2
// records, when present, are used as the evaluation state. If a dataset is
// edge-only, missing poses are initialized to identity rotation and zero
// translation so residual/Jacobian parity can still be benchmarked.
G2oDataset loadG2oSe2(const std::string& filename);

}  // namespace cora_symforce
