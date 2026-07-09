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
<div style="text-align:center; padding-top:8px; line-height:1.3;">

<p style="font-size:24px; font-weight:bold; margin:0;">
Developer
</p>

<p style="font-size:20px; margin:4px 0;">
Mohit Soni
</p>

<p style="font-size:18px; margin:4px 0;">
Master of Computer Applications (MCA)
</p>

<p style="font-size:18px; margin:4px 0;">
National Institute of Technology Kurukshetra
</p>

</div>
""",

    examples=[
        ["examples/normal.jpeg"],
        ["examples/pneumonia.jpeg"]
    ],

    flagging_mode="never"
)

demo.launch()