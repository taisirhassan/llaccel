#include "llaccel/device.h"

#ifdef LLACCEL_HAVE_RTL
#include "Vtop_v1.h"
#include "Vtop_v2.h"
#include "rtlsim_impl.h"
#endif

namespace llaccel {

std::unique_ptr<Device> makeRtlSim(const DeviceOptions& opt) {
#ifdef LLACCEL_HAVE_RTL
  if (opt.epilogueFusion) return std::make_unique<RtlSim<Vtop_v2>>(opt, "llaccel-v2, EPILOGUE_FUSION=1");
  return std::make_unique<RtlSim<Vtop_v1>>(opt, "llaccel-v1, EPILOGUE_FUSION=0");
#else
  (void)opt;
  throw std::runtime_error("this llaccel-sim was built without RTL support (LLACCEL_RTL=OFF)");
#endif
}

}  // namespace llaccel
