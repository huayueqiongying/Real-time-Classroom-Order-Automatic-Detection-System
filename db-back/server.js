const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;
const JWT_SECRET = 'your-secret-key'; // 生产环境中应该使用环境变量

// 中间件
app.use(cors());
app.use(express.json());

// 初始化数据库
const db = new sqlite3.Database('./users.db');

// 创建用户表
db.run(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

// 注册接口
app.post('/api/register', async (req, res) => {
  try {
    const { username, email, password } = req.body;

    // 验证输入
    if (!username || !email || !password) {
      return res.status(400).json({
        success: false,
        message: '用户名、邮箱和密码都是必填项'
      });
    }

    // 检查用户名是否已存在
    db.get('SELECT * FROM users WHERE username = ? OR email = ?', [username, email], async (err, user) => {
      if (err) {
        return res.status(500).json({
          success: false,
          message: '服务器错误'
        });
      }

      if (user) {
        return res.status(400).json({
          success: false,
          message: '用户名或邮箱已存在'
        });
      }

      // 加密密码
      const hashedPassword = await bcrypt.hash(password, 10);

      // 插入新用户
      db.run(
        'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
        [username, email, hashedPassword],
        function(err) {
          if (err) {
            return res.status(500).json({
              success: false,
              message: '注册失败'
            });
          }

          res.json({
            success: true,
            message: '注册成功！请登录',
            userId: this.lastID
          });
        }
      );
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: '服务器错误'
    });
  }
});

// 登录接口
app.post('/api/login', (req, res) => {
  try {
    const { username, password } = req.body;

    // 验证输入
    if (!username || !password) {
      return res.status(400).json({
        success: false,
        message: '用户名和密码都是必填项'
      });
    }

    // 查找用户
    db.get('SELECT * FROM users WHERE username = ?', [username], async (err, user) => {
      if (err) {
        return res.status(500).json({
          success: false,
          message: '服务器错误'
        });
      }

      if (!user) {
        return res.status(400).json({
          success: false,
          message: '用户名不存在'
        });
      }

      // 验证密码
      const isValidPassword = await bcrypt.compare(password, user.password);

      if (!isValidPassword) {
        return res.status(400).json({
          success: false,
          message: '密码错误'
        });
      }

      // 生成JWT令牌
      const token = jwt.sign(
        { userId: user.id, username: user.username },
        JWT_SECRET,
        { expiresIn: '24h' }
      );

      res.json({
        success: true,
        message: '登录成功',
        token,
        user: {
          id: user.id,
          username: user.username,
          email: user.email
        }
      });
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: '服务器错误'
    });
  }
});

// 验证token中间件
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({
      success: false,
      message: '访问令牌缺失'
    });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({
        success: false,
        message: '无效的访问令牌'
      });
    }

    req.user = user;
    next();
  });
};

// 获取用户信息接口（需要认证）
app.get('/api/user', authenticateToken, (req, res) => {
  db.get('SELECT id, username, email, created_at FROM users WHERE id = ?', [req.user.userId], (err, user) => {
    if (err) {
      return res.status(500).json({
        success: false,
        message: '服务器错误'
      });
    }

    if (!user) {
      return res.status(404).json({
        success: false,
        message: '用户不存在'
      });
    }

    res.json({
      success: true,
      user
    });
  });
});

// 获取所有用户接口（仅用于测试）
app.get('/api/users', (req, res) => {
  db.all('SELECT id, username, email, created_at FROM users', (err, users) => {
    if (err) {
      return res.status(500).json({
        success: false,
        message: '服务器错误'
      });
    }

    res.json({
      success: true,
      users
    });
  });
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`服务器运行在 http://localhost:${PORT}`);
  console.log('数据库已连接');
});

// 优雅关闭
process.on('SIGINT', () => {
  db.close((err) => {
    if (err) {
      console.error('关闭数据库时出错:', err.message);
    } else {
      console.log('数据库连接已关闭');
    }
    process.exit(0);
  });
});