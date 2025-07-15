const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const sqlite3 = require('sqlite3').verbose();

const router = express.Router();

// 先初始化数据库连接
const db = new sqlite3.Database('./user_data.db', (err) => {
  if (err) {
    console.error('数据库连接错误:', err);
  } else {
    console.log('已连接到SQLite数据库');
    // 初始化表
    db.run(`CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT UNIQUE NOT NULL,
      name TEXT NOT NULL,
      gender TEXT CHECK(gender IN ('male', 'female')) NOT NULL,
      phone TEXT NOT NULL,
      role TEXT CHECK(role IN ('teacher', 'student')) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )`, (err) => {
      if (err) {
        console.error('创建表失败:', err);
      } else {
        console.log('数据库表已就绪');
      }
    });
  }
});

// 然后配置multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, './public/uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    try {
      console.log('请求体内容:', req.body);
      const userId = req.body.userId;
      if (!userId) return cb(new Error('请提供学号或工号'));
      if (!/^[TS]\d+$/.test(userId)) {
        return cb(new Error('学号/工号格式不正确，应以S(学生)或T(老师)开头'));
      }
      const fileExt = path.extname(file.originalname).toLowerCase();
      cb(null, `${userId}${fileExt}`);
    } catch (err) {
      console.error('文件名生成错误:', err);
      cb(err);
    }
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];
    cb(null, allowedTypes.includes(file.mimetype));
  }
});

// 照片上传路由
router.post('/upload', upload.single('image'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: '未上传文件'
      });
    }
    res.json({
      success: true,
      path: `/uploads/${req.file.filename}`,
      userId: req.body.userId,
      message: '文件上传成功'
    });
  } catch (err) {
    res.status(500).json({
      success: false,
      message: err.message || '服务器内部错误'
    });
  }
});

// 添加请求体解析中间件
router.use(express.json());

// 用户信息保存路由
router.post('/save-user-data', (req, res) => {
  console.log('收到用户信息:', req.body);

  const { userId, name, gender, phone, role } = req.body;
  if (!userId || !name || !gender || !phone || !role) {
    return res.status(400).json({
      success: false,
      message: '缺少必要参数'
    });
  }

  db.run(
    `INSERT INTO users (user_id, name, gender, phone, role)
     VALUES (?, ?, ?, ?, ?)`,
    [userId, name, gender, phone, role],
    function(err) {
      if (err) {
        console.error('数据库错误:', err);
        return res.status(500).json({
          success: false,
          message: err.message.includes('UNIQUE') ?
                 '该用户已存在' : '数据库保存失败'
        });
      }
      console.log('用户信息已保存:', { userId, name });
      res.json({ success: true });
    }
  );
});
module.exports = router;