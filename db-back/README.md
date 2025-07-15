
# 1创建后端目录
mkdir db-back
cd db-back

# 2初始化项目
npm init -y

# 3安装依赖
# 3.1安装与Node.js v22兼容的sqlite3版本
npm install sqlite3@5.1.7 --sqlite3_binary_host_mirror=https://npmmirror.com/mirrors/sqlite3 --save
# 3.2
npm install express  bcrypt jsonwebtoken cors

# 4安装开发依赖
npm install -D nodemon
# 5创建后端文件

将  server.js 和 package.jsonbackend 文件保存到db-back目录
# 运行后端服务器：
node server.js
