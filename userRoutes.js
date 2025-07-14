const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const sqlite3 = require('sqlite3').verbose();
const router = express.Router();

// 数据库连接保持不变
const db = new sqlite3.Database('./user_data.db');

// 1. 获取用户信息API
router.get('/user/:id', (req, res) => {
  const userId = req.params.id;

  db.get(
    `SELECT * FROM users WHERE user_id = ?`,
    [userId],
    (err, row) => {
      if (err) {
        console.error('数据库查询错误:', err);
        return res.status(500).json({
          success: false,
          message: '数据库查询失败'
        });
      }

      if (!row) {
        return res.status(404).json({
          success: false,
          message: '用户不存在'
        });
      }

      res.json({
        success: true,
        data: {
          user_id: row.user_id,
          name: row.name,
          gender: row.gender,
          phone: row.phone,
          role: row.role,
          created_at: row.created_at
        }
      });
    }
  );
});

// 2. 更新用户信息API
router.put('/user/:id', (req, res) => {
  const userId = req.params.id;
  const { name, gender, phone } = req.body;

  // 验证输入
  if (!name || !gender || !phone) {
    return res.status(400).json({
      success: false,
      message: '缺少必要参数'
    });
  }

  db.run(
    `UPDATE users
     SET name = ?, gender = ?, phone = ?
     WHERE user_id = ?`,
    [name, gender, phone, userId],
    function(err) {
      if (err) {
        console.error('数据库更新错误:', err);
        return res.status(500).json({
          success: false,
          message: '数据库更新失败'
        });
      }

      if (this.changes === 0) {
        return res.status(404).json({
          success: false,
          message: '用户不存在'
        });
      }

      res.json({
        success: true,
        message: '用户信息更新成功'
      });
    }
  );
});

// 3. 文件上传配置保持不变
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../public/uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const userId = req.params.id;
    const fileExt = path.extname(file.originalname).toLowerCase();
    cb(null, `${userId}${fileExt}`);
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

// 4. 更新用户照片API
router.post('/user/:id/photo', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      message: '未上传文件'
    });
  }

  const photoPath = `/uploads/${req.file.filename}`;

  // 更新数据库中的照片路径
  db.run(
    `UPDATE users SET photo_path = ? WHERE user_id = ?`,
    [photoPath, req.params.id],
    (err) => {
      if (err) {
        console.error('数据库更新错误:', err);
        return res.status(500).json({
          success: false,
          message: '照片路径更新失败'
        });
      }

      res.json({
        success: true,
        path: photoPath,
        message: '照片上传成功'
      });
    }
  );
});

module.exports = router;