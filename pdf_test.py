import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from io import BytesIO

def generate_health_report_pdf(name):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica", 14)
    pdf.drawString(100, 800, f"Health Report for {name}")
    pdf.drawString(100, 770, "Diagnosis: Yes")
    pdf.drawString(100, 740, "Tips: Exercise, eat well")
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

st.title("🧪 PDF Download Test")

name = st.text_input("Your name")
if name:
    pdf = generate_health_report_pdf(name)
    st.download_button("📄 Download PDF", data=pdf, file_name="test_report.pdf", mime="application/pdf")
