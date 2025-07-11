<template>
  <div class="register-container">
    <h2>注册</h2>
    <form @submit.prevent="handleRegister">
      <div class="form-group">
        <label>用户名</label>
        <input v-model="username" type="text" required>
      </div>
      <div class="form-group">
        <label>邮箱</label>
        <input v-model="email" type="email" required>
      </div>
      <div class="form-group">
        <label>密码</label>
        <input v-model="password" type="password" required>
      </div>
      <div class="form-group">
        <label>确认密码</label>
        <input v-model="confirmPassword" type="password" required>
      </div>
      <button type="submit">注册</button>
      <p>已有账号？<router-link to="/login">立即登录</router-link></p>
    </form>
  </div>
</template>

<script>
export default {
  name: 'Register',
  data() {
    return {
      username: '',
      email: '',
      password: '',
      confirmPassword: ''
    }
  },
  methods: {
    handleRegister() {
      if (this.password !== this.confirmPassword) {
        alert('两次输入的密码不一致！')
        return
      }

      // 这里处理注册逻辑
      console.log('注册信息:', {
        username: this.username,
        email: this.email,
        password: this.password
      })

      // 实际项目中这里会调用API
      // this.$store.dispatch('user/register', { username, email, password })
      //   .then(() => {
      //     this.$router.push('/login')
      //   })
    }
  }
}
</script>

<style scoped>
.register-container {
  max-width: 400px;
  margin: 0 auto;
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 5px;
}
/* 复用登录页的样式 */
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
