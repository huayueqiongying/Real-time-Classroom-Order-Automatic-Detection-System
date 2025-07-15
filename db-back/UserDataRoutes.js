// userDataRoutes.js
const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const router = express.Router();

// 初始化师生信息数据库连接
const userDataDB = new sqlite3.Database('./user_data.db');

// 获取所有师生信息
router.get('/users', (req, res) => {
  userDataDB.all(`
    SELECT
      id,
      user_id,
      name,
      gender,
      phone,
      role,
      datetime(created_at, 'localtime') as created_at
    FROM users
    ORDER BY created_at DESC
  `, [], (err, rows) => {
    if (err) {
      console.error('数据库查询错误:', err);
      return res.status(500).json({
        success: false,
        message: '数据库查询失败'
      });
    }

    const users = rows.map(row => ({
      id: row.id,
      user_id: row.user_id || '',
      name: row.name || '',
      gender: row.gender || 'male',
      phone: row.phone || '',
      role: row.role || 'student',
      created_at: row.created_at || new Date().toISOString()
    }));

    res.json(users);
  });
});

// 在 userDataRoutes.js 中添加以下路由
router.delete('/users/:id', (req, res) => {
  const userId = req.params.id;

  if (!userId) {
    return res.status(400).json({
      success: false,
      message: '缺少用户ID参数'
    });
  }

  userDataDB.run('DELETE FROM users WHERE id = ?', [userId], function(err) {
    if (err) {
      console.error('删除用户错误:', err);
      return res.status(500).json({
        success: false,
        message: '删除用户失败'
      });
    }

    if (this.changes === 0) {
      return res.status(404).json({
        success: false,
        message: '未找到指定用户'
      });
    }

    res.json({
      success: true,
      message: '用户删除成功'
    });
  });
});

// 关闭数据库连接（当进程退出时）
process.on('SIGINT', () => {
  userDataDB.close();
});

module.exports = router;