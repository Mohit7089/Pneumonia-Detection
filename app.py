import gradio as gr
from predict import predict_image

demo = gr.Interface(
    fn=predict_image,

    inputs=gr.Image(
        type="filepath",
        label="📤 Upload Chest X-ray"
    ),

    outputs=gr.Label(
        num_top_classes=2,
        label="🩺 Prediction"
    ),

    title=" AI-Based Pneumonia Detection System",

    description="""
Upload a Chest X-ray image to predict whether the patient has **PNEUMONIA** or is **NORMAL**.

The model is built using **PyTorch** with a **MobileNetV2** backbone.
""",

    article="""
### Project Information

 **Developer:**  Mohit Soni

**Model:**  MobileNetV2

 **Framework:**  PyTorch

 **Test Accuracy:**  94%

 **Dataset:** Kaggle Chest X-ray Pneumonia Dataset
""",

    examples=[
        ["examples/normal.jpeg"],
        ["examples/pneumonia.jpeg"]
    ],

    flagging_mode="never"
)

demo.launch()