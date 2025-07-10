Vue前端部署在华为云116.205102.242:80上，后端在处理推流时需要from flask_cors import CORS，导入CORS，使得前端可以从云上访问后端处理的视频。
