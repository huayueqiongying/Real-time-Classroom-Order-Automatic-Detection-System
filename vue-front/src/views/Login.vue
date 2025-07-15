<template>
  <div class="login-container">
    <h2>登录</h2>
    <form @submit.prevent="handleLogin">
      <div class="form-group">
        <label>用户名</label>
        <input v-model="username" type="text" required :disabled="isLoading">
      </div>
      <div class="form-group">
        <label>密码</label>
        <input v-model="password" type="password" required :disabled="isLoading">
      </div>

      <!-- 滑块验证码组件 -->
      <SliderCaptcha
        ref="captcha"
        @success="onCaptchaSuccess"
        @error="onCaptchaError"
      />

      <!-- 错误提示 -->
      <div v-if="errorMessage" class="error-message">
        {{ errorMessage }}
      </div>

      <!-- 成功提示 -->
      <div v-if="successMessage" class="success-message">
        {{ successMessage }}
      </div>

      <button
        type="submit"
        :disabled="!isCaptchaValid || isLoading"
        :class="{ 'disabled': !isCaptchaValid || isLoading, 'loading': isLoading }"
      >
        {{ isLoading ? '登录中...' : '登录' }}
      </button>

      <p>还没有账号？<router-link to="/register">立即注册</router-link></p>
    </form>
  </div>
</template>

<script>
import SliderCaptcha from '@/components/SliderCaptcha'
import axios from 'axios'

export default {
  name: 'Login',
  components: {
    SliderCaptcha
  },
  data() {
    return {
      username: '',
      password: '',
      isCaptchaValid: false,
      isLoading: false,
      errorMessage: '',
      successMessage: ''
    }
  },
  methods: {
    onCaptchaSuccess() {
      this.isCaptchaValid = true
      console.log('验证码验证成功')
    },

    onCaptchaError() {
      this.isCaptchaValid = false
      console.log('验证码验证失败')
    },

    async handleLogin() {
      if (!this.isCaptchaValid) {
        this.errorMessage = '请先完成滑块验证'
        return
      }

      // 清空之前的消息
      this.errorMessage = ''
      this.successMessage = ''
      this.isLoading = true

      try {
        // 调用后端API
        const response = await axios.post('http://localhost:3000/api/login', {
          username: this.username,
          password: this.password
        })

        if (response.data.success) {
          this.successMessage = response.data.message

          // 保存用户信息和token
          localStorage.setItem('token', response.data.token)
          localStorage.setItem('user', JSON.stringify(response.data.user))

          // 更新父组件的登录状态
          this.$parent.isAuthenticated = true

          // 跳转到首页
          setTimeout(() => {
            this.$router.push('/dashboard')
          }, 1000)
        }
      } catch (error) {
        if (error.response && error.response.data) {
          this.errorMessage = error.response.data.message
        } else {
          this.errorMessage = '登录失败，请检查网络连接'
        }

        // 重置验证码
        this.$refs.captcha.reset()
        this.isCaptchaValid = false
      } finally {
        this.isLoading = false
      }
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
  margin-top: 10px;
  transition: background-color 0.2s;
}

button:hover:not(.disabled) {
  background-color: #369f6e;
}

button.disabled {
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
