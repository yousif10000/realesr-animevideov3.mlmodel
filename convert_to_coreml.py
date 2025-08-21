import torch
import coremltools as ct
from basicsr.archs.srvgg_arch import SRVGGNetCompact

# 1. بناء النموذج
model = SRVGGNetCompact(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_conv=16,
    upscale=4,
    act_type="prelu"
)

# 2. تحميل الأوزان
ckpt = torch.load("models/realesr-animevideov3.pth", map_location="cpu")
if "params" in ckpt:
    ckpt = ckpt["params"]
model.load_state_dict(ckpt, strict=True)
model.eval()

# 3. dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# 4. التحويل
mlmodel = ct.convert(
    model,
    source="pytorch",
    inputs=[ct.ImageType(
        name="input",
        shape=dummy_input.shape,
        scale=1/255.0,
        bias=[0,0,0],
        color_layout="RGB"
    )],
    outputs=[ct.ImageType(name="output")],
    minimum_deployment_target=ct.target.iOS16
)

# 5. حفظ
mlmodel.save("realesr-animevideov3.mlmodel")
print("✅ realesr-animevideov3.mlmodel جاهز")
