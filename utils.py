# import streamlit as st
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import A4
# from io import BytesIO

# def generate_health_report_pdf(name):
#     buffer = BytesIO()
#     pdf = canvas.Canvas(buffer, pagesize=A4)
#     pdf.setFont("Helvetica", 14)
#     pdf.drawString(100, 800, f"Health Report for {name}")
#     pdf.drawString(100, 770, "Diagnosis: Yes")
#     pdf.drawString(100, 740, "Tips: Exercise, eat well")
#     pdf.save()
#     buffer.seek(0)
#     return buffer.getvalue()

# st.title("🧪 PDF Download Test")

# name = st.text_input("Your name")
# if name:
#     pdf = generate_health_report_pdf(name)
#     st.download_button("📄 Download PDF", data=pdf, file_name="test_report.pdf", mime="application/pdf")


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

# 👇 This block will only run if you open utils.py directly (for testing)
if __name__ == "__main__":
    st.title("🧪 PDF Download Test")

    name = st.text_input("Your name")
    if name:
        tips = ["Stay active", "Eat healthy", "Sleep well"]
        diet = "Day 1:\n- Breakfast: Oats\n- Lunch: Grilled vegetables\n- Dinner: Soup"
        pdf = generate_health_report_pdf(name, "Test Condition", "Yes", tips, diet)

        st.download_button("📄 Download PDF", data=pdf, file_name="test_report.pdf", mime="application/pdf")
