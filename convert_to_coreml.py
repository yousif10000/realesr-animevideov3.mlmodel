import torch
import coremltools as ct
import numpy as np

# تحميل النموذج مباشرة
model = torch.load("models/realesr-animevideov3.pth", map_location="cpu")
model.eval()

# تحويل النموذج
dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, dummy_input)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=dummy_input.shape)],
    minimum_deployment_target=ct.target.iOS16
)

mlmodel.save("converted_model.mlmodel")
