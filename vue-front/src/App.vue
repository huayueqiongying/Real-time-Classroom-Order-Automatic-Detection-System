<template>
  <div id="app">
    <!-- 导航栏只在不登录/注册页面显示 -->
    <div v-if="showNavigation" class="nav-container">
      <router-link to="/">首页</router-link>
      <router-link to="/about">关于</router-link>
      <router-link to="/login" v-if="!isAuthenticated">登录</router-link>
      <router-link to="/register" v-if="!isAuthenticated">注册</router-link>
      <a href="#" @click.prevent="logout" v-if="isAuthenticated">退出</a>
    </div>

    <!-- 主内容区域 -->
    <router-view/>

    <!-- 移除原测试内容 -->
  </div>
</template>

<script>
import HelloWorld from './components/HelloWorld'

export default {
  name: 'App',
  components: {
    HelloWorld
  },
  data() {
    return {
      title: 'Vue CLI 2.x 项目',
      count: 0,
      isAuthenticated: false // 根据实际登录状态更新
    }
  },
  computed: {
    showNavigation() {
      // 登录和注册页面不显示导航
      return !['/login', '/register'].includes(this.$route.path)
    }
  },
  methods: {
    changeTitle() {
      this.title = '标题已更新!'
    },
    increment() {
      this.count++
    },
    logout() {
      // 处理退出逻辑
      this.isAuthenticated = false
      this.$router.push('/login')
    }
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  color: #2c3e50;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

/* 新增导航栏样式 */
.nav-container {
  padding: 15px;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  margin-bottom: 30px;
  border-radius: 8px;
  display: flex;
  justify-content: center;
  gap: 15px;
}

/* 更新导航链接样式 */
a {
  color: #34495e;
  text-decoration: none;
  font-weight: 500;
  padding: 8px 15px;
  border-radius: 4px;
  transition: all 0.2s;
}

a:hover {
  background-color: #f5f5f5;
  color: #42b983;
}

/* 更新按钮样式 */
button {
  padding: 10px 20px;
  margin: 0 5px;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.3s;
}

button:hover {
  background-color: #369f6e;
}

/* 登录/注册页面容器样式 */
.auth-container {
  max-width: 400px;
  margin: 30px auto;
  padding: 25px;
  border: 1px solid #eaeaea;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  background: white;
}

/* 移除原有的竖线分隔符，改用gap间距 */
</style>
