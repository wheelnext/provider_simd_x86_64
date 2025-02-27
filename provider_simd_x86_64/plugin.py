import os

from variantlib.config import KeyConfig
from variantlib.config import ProviderConfig

from provider_simd_x86_64 import __version__

FLAGS = """
3dnow 3dnowext aes avx avx2 avx512_4fmaps avx512_4vnniw avx512_bf16
avx512_bitalg avx512_fp16 avx512_vbmi2 avx512_vnni avx512_vp2intersect
avx512_vpopcntdq avx512bw avx512cd avx512dq avx512er avx512f avx512ifma
avx512pf avx512vbmi avx512vl f16c fma3 fma4 mmx mmxext padlock pclmul popcnt
rdrand sha sse sse2 sse3 sse4_1 sse4_2 sse4a ssse3 vpclmulqdq xop
""".split()


class SIMD_X86_64_Plugin:  # noqa: N801
    __provider_name__ = "provider_simd_x86_64"
    __version__ = __version__

    def _get_capabilities(self):
        stop_after = int(os.environ.get("SIMD_STOP_AFTER", "1000"))

        for i, simd in enumerate(FLAGS):
            if i == stop_after:
                break
            yield KeyConfig(key=simd, values=["1"])

    def run(self) -> ProviderConfig:
        keyconfigs = list(self._get_capabilities())

        return ProviderConfig(provider=self.__provider_name__, configs=keyconfigs)
