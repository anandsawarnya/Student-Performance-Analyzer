# Student Performance Analyzer — Premium Dark UI

A Streamlit website for the HCL-Tech guided academic project. It uses real ML models trained from the CSV supplied by the user.

## Included
- Premium dark navy dashboard
- Grade prediction form connected to a trained model
- Analytics: grade distribution, subject averages, attendance and study-hours charts
- Model comparison: Logistic Regression, Decision Tree, Random Forest
- Accuracy, weighted precision, recall, F1-score and confusion matrix
- CSV upload and cleaned dataset download

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Put `Student_Performance.csv` beside `app.py`, **or** launch the app and upload the CSV from the sidebar.
5. Run:

   ```bash
   streamlit run app.py
   ```

6. Open the local address Streamlit shows in your terminal.

## Dataset requirements
- CSV must include `final_grade` as the target column (grades such as A–F).
- The app follows the SRS by excluding `overall_score` from model inputs.
- The expected SRS columns include `Age`, `Gender`, `Schoool_types`, `parent_education`, `study_hours`, `attendance_percentage`, `internet_access`, `travel_time`, `extra_activities`, `study_method`, `math_score`, `science_score`, `english_score`, `overall_score`, and `final_grade`.
- Some common header spelling variants are handled automatically.

## Notes
- No fake/demo prediction is returned if no dataset is supplied.
- Metrics are recalculated from the uploaded dataset and may differ from values quoted in the SRS.
- This is an academic prototype. Predicted grades are estimates, not official academic decisions.
