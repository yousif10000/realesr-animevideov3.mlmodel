import sys
import coremltools as ct

if len(sys.argv) < 3:
    print("Usage: python convert_to_coreml.py <input.onnx> <output.mlmodel>")
    sys.exit(1)

onnx_model_path = sys.argv[1]
mlmodel_path = sys.argv[2]

# التحويل (بدون source="onnx")
mlmodel = ct.converters.onnx.convert(
    model=onnx_model_path,
    minimum_deployment_target=ct.target.iOS16,
    inputs=[
        ct.ImageType(
            name="input",
            shape=(1, 3, ct.RangeDim(16, 4096), ct.RangeDim(16, 4096)),
            scale=1/255.0,
            bias=[0, 0, 0],
            color_layout="RGB"
        )
    ],
    outputs=[ct.ImageType(name="output")]
)

mlmodel.save(mlmodel_path)
print(f"✅ تم التحويل: {onnx_model_path} → {mlmodel_path}")
