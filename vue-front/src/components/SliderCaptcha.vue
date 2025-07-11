<template>
  <div class="slider-captcha">
    <div class="captcha-container">
      <div class="captcha-bg" :style="{ backgroundImage: `url(${bgImage})` }">
        <div class="captcha-puzzle" :style="puzzleStyle"></div>
      </div>
      <div class="captcha-refresh" @click="refreshCaptcha">
        <span>🔄</span>
      </div>
    </div>

    <div class="slider-container">
      <div class="slider-track" :class="{ 'success': isSuccess, 'error': isError }">
        <div class="slider-fill" :style="{ width: sliderPosition + 'px' }"></div>
        <div
          class="slider-button"
          :style="{ left: sliderPosition + 'px' }"
          :class="{ 'dragging': isDragging }"
          @mousedown="startDrag"
          @touchstart="startDrag"
        >
          <span v-if="!isSuccess && !isError">→</span>
          <span v-if="isSuccess">✓</span>
          <span v-if="isError">✗</span>
        </div>
      </div>
      <div class="slider-text">
        <span v-if="!isSuccess && !isError">拖动滑块完成验证</span>
        <span v-if="isSuccess" class="success-text">验证成功</span>
        <span v-if="isError" class="error-text">验证失败，请重试</span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SliderCaptcha',
  data() {
    return {
      bgImage: '',
      puzzlePosition: 0,
      sliderPosition: 0,
      isDragging: false,
      isSuccess: false,
      isError: false,
      startX: 0,
      tolerance: 10, // 允许的误差范围
      maxSliderPosition: 0
    }
  },
  computed: {
    puzzleStyle() {
      // 拼图块从左边开始，跟随滑块移动
      return {
        left: this.sliderPosition + 'px',
        backgroundImage: `url(${this.bgImage})`,
        backgroundPosition: `-${this.puzzlePosition}px 0px`
      }
    }
  },
  mounted() {
    this.initCaptcha()
    this.maxSliderPosition = 260 // 滑块轨道宽度 - 滑块宽度
  },
  methods: {
    initCaptcha() {
      // 先生成拼图位置，再生成背景图片
      this.puzzlePosition = Math.random() * 200 + 50 // 50-250px之间
      this.generateBackground()
      this.resetSlider()
    },

    generateBackground() {
      // 创建一个简单的渐变背景图片
      const canvas = document.createElement('canvas')
      canvas.width = 300
      canvas.height = 150
      const ctx = canvas.getContext('2d')

      // 创建渐变背景
      const gradient = ctx.createLinearGradient(0, 0, 300, 150)
      gradient.addColorStop(0, '#667eea')
      gradient.addColorStop(1, '#764ba2')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, 300, 150)

      // 添加一些随机圆点装饰
      for (let i = 0; i < 20; i++) {
        ctx.beginPath()
        ctx.arc(
          Math.random() * 300,
          Math.random() * 150,
          Math.random() * 5 + 2,
          0,
          2 * Math.PI
        )
        ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.5 + 0.1})`
        ctx.fill()
      }

      // 在拼图位置绘制缺口轮廓
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.strokeRect(this.puzzlePosition, 50, 50, 50)

      this.bgImage = canvas.toDataURL()
    },

    refreshCaptcha() {
      this.initCaptcha()
    },

    startDrag(e) {
      if (this.isSuccess) return

      this.isDragging = true
      this.isError = false
      this.startX = e.type === 'mousedown' ? e.clientX : e.touches[0].clientX

      document.addEventListener('mousemove', this.onDrag)
      document.addEventListener('mouseup', this.endDrag)
      document.addEventListener('touchmove', this.onDrag)
      document.addEventListener('touchend', this.endDrag)
    },

    onDrag(e) {
      if (!this.isDragging) return

      const currentX = e.type === 'mousemove' ? e.clientX : e.touches[0].clientX
      const deltaX = currentX - this.startX

      this.sliderPosition = Math.max(0, Math.min(this.maxSliderPosition, deltaX))
    },

    endDrag() {
      if (!this.isDragging) return

      this.isDragging = false
      document.removeEventListener('mousemove', this.onDrag)
      document.removeEventListener('mouseup', this.endDrag)
      document.removeEventListener('touchmove', this.onDrag)
      document.removeEventListener('touchend', this.endDrag)

      this.verifyCaptcha()
    },

    verifyCaptcha() {
      // 验证滑块位置是否接近拼图缺口位置
      const difference = Math.abs(this.sliderPosition - this.puzzlePosition)

      if (difference <= this.tolerance) {
        this.isSuccess = true
        this.isError = false
        this.$emit('success')
      } else {
        this.isError = true
        this.isSuccess = false
        this.$emit('error')

        // 2秒后重置
        setTimeout(() => {
          this.resetSlider()
        }, 2000)
      }
    },

    resetSlider() {
      this.sliderPosition = 0
      this.isSuccess = false
      this.isError = false
      this.isDragging = false
    },

    // 公共方法：重置验证码
    reset() {
      this.initCaptcha()
    }
  }
}
</script>

<style scoped>
.slider-captcha {
  margin: 20px 0;
}

.captcha-container {
  position: relative;
  width: 300px;
  height: 150px;
  margin: 0 auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  overflow: hidden;
}

.captcha-bg {
  width: 100%;
  height: 100%;
  background-size: cover;
  background-position: center;
}

.captcha-puzzle {
  position: absolute;
  top: 50px;
  width: 50px;
  height: 50px;
  background-size: 300px 150px;
  border: 2px solid #fff;
  border-radius: 4px;
  box-shadow: 0 0 10px rgba(0,0,0,0.3);
}

.captcha-refresh {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 30px;
  height: 30px;
  background: rgba(255,255,255,0.8);
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.captcha-refresh:hover {
  background: rgba(255,255,255,0.9);
}

.slider-container {
  width: 300px;
  margin: 20px auto 0;
}

.slider-track {
  position: relative;
  width: 100%;
  height: 40px;
  background: #f0f0f0;
  border-radius: 20px;
  border: 1px solid #ddd;
  overflow: hidden;
}

.slider-track.success {
  background: #e8f5e8;
  border-color: #4caf50;
}

.slider-track.error {
  background: #ffeaea;
  border-color: #f44336;
}

.slider-fill {
  height: 100%;
  background: linear-gradient(90deg, #4caf50, #81c784);
  border-radius: 20px;
  transition: width 0.2s;
}

.slider-track.error .slider-fill {
  background: linear-gradient(90deg, #f44336, #ef5350);
}

.slider-button {
  position: absolute;
  top: 0;
  width: 40px;
  height: 40px;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  transition: all 0.2s;
  user-select: none;
}

.slider-button:hover {
  border-color: #42b983;
  transform: scale(1.1);
}

.slider-button.dragging {
  transform: scale(1.1);
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.slider-text {
  text-align: center;
  margin-top: 10px;
  font-size: 14px;
  color: #666;
}

.success-text {
  color: #4caf50;
}

.error-text {
  color: #f44336;
}
</style>
