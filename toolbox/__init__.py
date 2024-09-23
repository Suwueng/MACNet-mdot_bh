import os
from dotenv import load_dotenv

import toolbox.dataload
import toolbox.dataplot
import toolbox.createdataset

load_dotenv()  # 加载 .env 文件中的环境变量
database_path = os.getenv("DATABASE_PATH")
root_path = os.getenv("ROOT_PATH")
