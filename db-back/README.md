1. 创建后端项目
bash# 创建后端目录
mkdir backend
cd backend

# 初始化项目
npm init -y

# 安装依赖
npm install express sqlite3 bcrypt jsonwebtoken cors

# 安装开发依赖
npm install -D nodemon
2. 创建后端文件

将  server.js 和 package.jsonbackend 文件保存到db-back目录
运行后端服务器：
npm start
# 或者开发模式：
npm run dev
