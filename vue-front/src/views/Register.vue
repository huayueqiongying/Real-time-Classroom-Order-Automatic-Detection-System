<template>
  <div class="register-container">
    <h2>注册</h2>
    <form @submit.prevent="handleRegister">
      <div class="form-group">
        <label>用户名</label>
        <input v-model="username" type="text" required :disabled="isLoading">
      </div>
      <div class="form-group">
        <label>邮箱</label>
        <input v-model="email" type="email" required :disabled="isLoading">
      </div>
      <div class="form-group">
        <label>密码</label>
        <input v-model="password" type="password" required :disabled="isLoading">
      </div>
      <div class="form-group">
        <label>确认密码</label>
        <input v-model="confirmPassword" type="password" required :disabled="isLoading">
      </div>

      <!-- 错误提示 -->
      <div v-if="errorMessage" class="error-message">
        {{ errorMessage }}
      </div>

      <!-- 成功提示 -->
      <div v-if="successMessage" class="success-message">
        {{ successMessage }}
      </div>

      <button type="submit" :disabled="isLoading" :class="{ 'loading': isLoading }">
        {{ isLoading ? '注册中...' : '注册' }}
      </button>
      <p>已有账号？<router-link to="/login">立即登录</router-link></p>
    </form>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'Register',
  data() {
    return {
      username: '',
      email: '',
      password: '',
      confirmPassword: '',
      isLoading: false,
      errorMessage: '',
      successMessage: ''
    }
  },
  methods: {
    async handleRegister() {
      // 清空之前的消息
      this.errorMessage = ''
      this.successMessage = ''

      // 前端验证
      if (this.password !== this.confirmPassword) {
        this.errorMessage = '两次输入的密码不一致！'
        return
      }

      if (this.password.length < 6) {
        this.errorMessage = '密码长度至少6位！'
        return
      }

      this.isLoading = true

      try {
        // 调用后端API
        const response = await axios.post('http://localhost:3000/api/register', {
          username: this.username,
          email: this.email,
          password: this.password
        })

        if (response.data.success) {
          this.successMessage = response.data.message

          // 清空表单
          this.username = ''
          this.email = ''
          this.password = ''
          this.confirmPassword = ''

          // 3秒后跳转到登录页
          setTimeout(() => {
            this.$router.push('/login')
          }, 3000)
        }
      } catch (error) {
        if (error.response && error.response.data) {
          this.errorMessage = error.response.data.message
        } else {
          this.errorMessage = '注册失败，请检查网络连接'
        }
      } finally {
        this.isLoading = false
      }
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
  border: 1px solid #ddd;
  border-radius: 4px;
}

input:disabled {
  background-color: #f5f5f5;
  cursor: not-allowed;
}

button {
  width: 100%;
  padding: 10px;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

button:hover:not(:disabled) {
  background-color: #369f6e;
}

button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

button.loading {
  background-color: #42b983;
  opacity: 0.7;
}

.error-message {
  background-color: #ffeaea;
  color: #d32f2f;
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 15px;
  border: 1px solid #ffcdd2;
}

.success-message {
  background-color: #e8f5e8;
  color: #388e3c;
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 15px;
  border: 1px solid #c8e6c9;
}

a {
  color: #42b983;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

p {
  text-align: center;
  margin-top: 15px;
}
</style>
