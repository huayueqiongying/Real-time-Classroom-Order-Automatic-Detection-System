<template>
  <div class="login-container">
    <h2>登录</h2>
    <form @submit.prevent="handleLogin">
      <div class="form-group">
        <label>用户名</label>
        <input v-model="username" type="text" required>
      </div>
      <div class="form-group">
        <label>密码</label>
        <input v-model="password" type="password" required>
      </div>
      <button type="submit">登录</button>
      <p>还没有账号？<router-link to="/register">立即注册</router-link></p>
    </form>
  </div>
</template>

<script>
export default {
  name: 'Login',
  data() {
    return {
      username: '',
      password: ''
    }
  },
  methods: {
    handleLogin() {
      // 这里处理登录逻辑
      console.log('登录信息:', {
        username: this.username,
        password: this.password
      })

      // 实际项目中这里会调用API
      // this.$store.dispatch('user/login', { username, password })
      //   .then(() => {
      //     this.$router.push('/dashboard')
      //   })
    }
  }
}
</script>

<style scoped>
.login-container {
  max-width: 400px;
  margin: 0 auto;
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 5px;
}
.form-group {
  margin-bottom: 15px;
}
label {
  display: block;
  margin-bottom: 5px;
}
input {
  width: 100%;
  padding: 8px;
  box-sizing: border-box;
}
button {
  width: 100%;
  padding: 10px;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
</style>
