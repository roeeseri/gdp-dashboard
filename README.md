# Spotify Israel – Listening Patterns Around 7.10 (Streamlit Dashboard)

A Streamlit dashboard analyzing Israeli Spotify listening patterns **before and after Oct 7, 2023**.
The app is built from:
- `data/merged_all_weeks.csv` (weekly chart data + audio features)
- `artist_meta.xlsx` (artist groups + optional image paths)
- optional `artists_photos/` folder (artist avatars)

---

[![Open in Streamlit](https://gdp-dashboard-mo6fq9uiz09.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
