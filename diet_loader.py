import pandas as pd

def load_diet_plan(disease: str, days: int = 3, file_path: str = "diet_plans.xlsx") -> str:
    try:
        df = pd.read_excel(file_path)
        df = df[df["Disease"].str.lower() == disease.lower()]
        df = df[df["Day"] <= days]

        if df.empty:
            return f"No diet plan found for {disease.title()} for {days} days."

        output = f"### 🥗 Diet Plan for {disease.title()} – {days} Day(s)\n"
        grouped = df.groupby("Day")

        for day, meals in grouped:
            output += f"\n**Day {day}**\n"
            for _, row in meals.iterrows():
                output += f"- {row['Meal Type']}: {row['Meal Description']}\n"
        return output

    except Exception as e:
        return f"⚠️ Error loading diet plan: {e}"