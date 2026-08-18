import argparse
import logging
import time
import schedule
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

def run_pipeline():
    """Runs the full data collection, cleaning, feature engineering, and model training pipeline."""
    logger.info("=== Starting Data Pipeline ===")
    
    try:
        logger.info("1. Collecting Data...")
        subprocess.run([sys.executable, "-m", "src.collector"], check=True)
        
        logger.info("2. Cleaning Data...")
        subprocess.run([sys.executable, "-m", "src.cleaner"], check=True)
        
        logger.info("3. Engineering Features...")
        subprocess.run([sys.executable, "-m", "src.feature_engineering"], check=True)
        
        logger.info("4. Training Models...")
        subprocess.run([sys.executable, "-m", "src.train"], check=True)
        
        logger.info("=== Pipeline Completed Successfully ===")
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed at step: {e.cmd}")
        
def run_dashboard():
    """Launches the Streamlit dashboard."""
    logger.info("Launching Streamlit Dashboard...")
    dashboard_path = BASE_DIR / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])

def run_daemon(interval_hours: int):
    """Runs the data pipeline continuously on a schedule."""
    logger.info(f"Starting continuous background job. Pipeline will run every {interval_hours} hours.")
    # Run once immediately
    run_pipeline()
    
    schedule.every(interval_hours).hours.do(run_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Content Intelligence CLI")
    parser.add_argument("--collect", action="store_true", help="Run the full data pipeline once.")
    parser.add_argument("--dashboard", action="store_true", help="Launch the Streamlit dashboard.")
    parser.add_argument("--daemon", type=int, metavar="HOURS", help="Run continuously in the background every N hours.")
    
    args = parser.parse_args()
    
    if args.daemon:
        run_daemon(args.daemon)
    elif args.collect:
        run_pipeline()
    elif args.dashboard:
        run_dashboard()
    else:
        parser.print_help()
