<template>
  <div id="app">
    <!-- 导航栏只在不登录/注册页面显示 -->
    <div v-if="showNavigation">
      <router-link to="/">首页</router-link> |
      <router-link to="/about">关于</router-link> |
      <router-link to="/login" v-if="!isAuthenticated">登录</router-link>
      <template v-if="!isAuthenticated"> | </template>
      <router-link to="/register" v-if="!isAuthenticated">注册</router-link>
      <a href="#" @click.prevent="logout" v-if="isAuthenticated">退出</a>
    </div>

    <!-- 主内容区域 -->
    <router-view/>

    <!-- 原测试内容可以保留或移除 -->
    <template v-if="$route.path === '/'">
      <h1>{{ title }}</h1>
      <img src="./assets/logo.png">
      <button @click="changeTitle">修改标题</button>
      <p>计数器: {{ count }}</p>
      <button @click="increment">+1</button>
      <HelloWorld msg="这是子组件"/>
    </template>
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
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

button {
  padding: 8px 16px;
  margin: 0 5px;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button:hover {
  background-color: #369f6e;
}

/* 导航链接样式 */
a {
  color: #42b983;
  text-decoration: none;
  margin: 0 5px;
}

a:hover {
  text-decoration: underline;
}

/* 登录/注册页面容器样式 */
.auth-container {
  max-width: 400px;
  margin: 30px auto;
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
