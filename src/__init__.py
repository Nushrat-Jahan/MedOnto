from dotenv import load_dotenv
import warnings

load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*")