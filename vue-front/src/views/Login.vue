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

      <!-- 滑块验证码组件 -->
      <SliderCaptcha
        ref="captcha"
        @success="onCaptchaSuccess"
        @error="onCaptchaError"
      />

      <button
        type="submit"
        :disabled="!isCaptchaValid"
        :class="{ 'disabled': !isCaptchaValid }"
      >
        登录
      </button>

      <p>还没有账号？<router-link to="/register">立即注册</router-link></p>
    </form>
  </div>
</template>

<script>
import SliderCaptcha from '@/components/SliderCaptcha'

export default {
  name: 'Login',
  components: {
    SliderCaptcha
  },
  data() {
    return {
      username: '',
      password: '',
      isCaptchaValid: false
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

    handleLogin() {
      if (!this.isCaptchaValid) {
        alert('请先完成滑块验证')
        return
      }

      // 这里处理登录逻辑
      console.log('登录信息:', {
        username: this.username,
        password: this.password
      })

      // 模拟登录成功
      // 实际项目中这里会调用API
      alert('登录成功！')

      // 登录成功后可以跳转到主页
      // this.$router.push('/home')

      // 或者更新父组件的登录状态
      // this.$parent.isAuthenticated = true

      // 重置验证码以便下次登录
      this.$refs.captcha.reset()
      this.isCaptchaValid = false
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

button.disabled:hover {
  background-color: #ccc;
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
