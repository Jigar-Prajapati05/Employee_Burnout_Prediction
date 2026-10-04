import pandas as pd
import joblib

model_pipeline = joblib.load("burnout_pipeline.pkl")

print("Model loaded successfully!")

person_data = {
    "Age": 32,
    "Gender": "Male",
    "Marital_Status": "Married",
    "Experience_Years": 8,
    "Working_Hours_per_Week": 55,
    "Commute_Time_Hours": 1.5,
    "Remote_Work": "No",
    "Stress_Level": "High",
    "Health_Issues": "Unknown",
    "Company_Size": "Large",
    "Department": "Operations",
    "Sleep_Hours": 5,
    "Physical_Activity_Hours_per_Week": 2,
    "Mental_Health_Leave_Taken": "No",
    "Manager_Support_Level": "Low",
    "Work_Pressure_Level": "High",
    "Annual_Leaves_Taken": 6,
    "Work_Life_Balance": "Low",
    "Family_Support_Level": "Medium",
    "Job_Satisfaction": "Low",
    "Performance_Rating": "Good",
    "Team_Size": 15,
    "Training_Opportunities": "Yes",
    "Gender_Bias_Experienced": "No",
    "Discrimination_Experienced": "No",
    "Location": "Delhi"
}


person_df = pd.DataFrame([person_data])

# print(person_df)

prediction = model_pipeline.predict(person_df)

# print("Predicted Burnout Symptoms:", prediction[0])
probabilities = model_pipeline.predict_proba(person_df)[0]
classes = model_pipeline.classes_


print("\n========================================")
print("       PERSONNEL ML ASSESSMENT")
print("========================================")

print(f"\nPredicted Class: {prediction[0].upper()}")

print("\nEstimated Class Probabilities:")

for cls, prob in zip(classes, probabilities):
    print(f"  {cls:<10}: {prob:.1%}")
    

print("\nObserved indicators:")

if person_df["Working_Hours_per_Week"].iloc[0] >= 50:
    print("- High weekly working hours")

if person_df["Sleep_Hours"].iloc[0] <= 6:
    print("- Low sleep duration")

if person_df["Stress_Level"].iloc[0] == "High":
    print("- High stress level")

if person_df["Work_Pressure_Level"].iloc[0] == "High":
    print("- High work pressure")

if person_df["Work_Life_Balance"].iloc[0] == "Low":
    print("- Low work-life balance")

if person_df["Manager_Support_Level"].iloc[0] == "Low":
    print("- Low manager support")

if person_df["Job_Satisfaction"].iloc[0] == "Low":
    print("- Low job satisfaction")

print("\n========================================")

'''Are these the reasons for the prediction?
    They are notable indicators in the input profile. The model itself makes the prediction using the complete feature set'''