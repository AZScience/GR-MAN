# Entry point for Streamlit Cloud deployment
import runpy

if __name__ == "__main__":
    runpy.run_path("app_medical_pipeline.py", run_name="__main__")
