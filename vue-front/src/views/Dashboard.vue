<template>
  <div class="dashboard">
    <!-- 顶部标题栏 -->
    <div class="header">
      <img src="@/assets/xiaohui.png" alt="校徽" class="logo">
      <h1>智能教室监控系统</h1>
    </div>

    <!-- 功能网格 -->
    <div class="function-grid">
      <router-link to="/camera" class="function-card">
        <div class="card-icon">📷</div>
        <h3>查看摄像头</h3>
        <p>实时监控教室情况</p>
      </router-link>

      <router-link to="/events" class="function-card">
        <div class="card-icon">⚠️</div>
        <h3>异常事件处理</h3>
        <p>处理教室异常情况</p>
      </router-link>

      <router-link to="/profile" class="function-card">
        <div class="card-icon">👤</div>
        <h3>添加个人信息</h3>
        <p>查看和修改个人信息</p>
      </router-link>

      <router-link to="/students" class="function-card">
        <div class="card-icon">👥</div>
        <h3>查看名单</h3>
        <p>管理班级学生信息</p>
      </router-link>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Dashboard'
}
</script>

<style scoped>
.dashboard {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 40px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.logo {
  height: 60px;
  margin-right: 20px;
}

h1 {
  color: #2c3e50;
  font-size: 28px;
  margin: 0;
}

.function-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 25px;
}

.function-card {
  padding: 25px;
  border-radius: 8px;
  background: white;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  text-align: center;
  text-decoration: none;
  color: #2c3e50;
  transition: all 0.3s ease;
}

.function-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.card-icon {
  font-size: 40px;
  margin-bottom: 15px;
}

h3 {
  margin: 0 0 10px 0;
  color: #34495e;
}

p {
  margin: 0;
  color: #7f8c8d;
  font-size: 14px;
}
</style>
