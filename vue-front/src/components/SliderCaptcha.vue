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
      maxSliderPosition: 0,
      currentPuzzleType: 'square', // 当前拼图类型
      puzzleTypes: ['square', 'circle', 'triangle', 'star', 'heart', 'diamond'], // 可用的拼图类型
      backgroundTypes: ['gradient1', 'gradient2', 'pattern1', 'pattern2', 'geometric', 'dots'] // 背景类型
    }
  },
  computed: {
    puzzleStyle() {
      return {
        left: this.sliderPosition + 'px',
        backgroundImage: `url(${this.bgImage})`,
        backgroundPosition: `-${this.puzzlePosition}px 0px`,
        maskImage: `url(${this.getPuzzleMask()})`,
        WebkitMaskImage: `url(${this.getPuzzleMask()})`,
        maskSize: '50px 50px',
        WebkitMaskSize: '50px 50px',
        maskRepeat: 'no-repeat',
        WebkitMaskRepeat: 'no-repeat'
      }
    }
  },
  mounted() {
    this.initCaptcha()
    this.maxSliderPosition = 260 // 滑块轨道宽度 - 滑块宽度
  },
  methods: {
    initCaptcha() {
      // 随机选择拼图类型和背景类型
      this.currentPuzzleType = this.puzzleTypes[Math.floor(Math.random() * this.puzzleTypes.length)]
      this.puzzlePosition = Math.random() * 200 + 50 // 50-250px之间
      this.generateBackground()
      this.resetSlider()
    },

    generateBackground() {
      const canvas = document.createElement('canvas')
      canvas.width = 300
      canvas.height = 150
      const ctx = canvas.getContext('2d')

      // 随机选择背景类型
      const bgType = this.backgroundTypes[Math.floor(Math.random() * this.backgroundTypes.length)]

      switch(bgType) {
        case 'gradient1':
          this.createGradientBackground(ctx, ['#667eea', '#764ba2'])
          break
        case 'gradient2':
          this.createGradientBackground(ctx, ['#ff9a9e', '#fecfef'])
          break
        case 'pattern1':
          this.createPatternBackground(ctx, '#4facfe', '#00f2fe')
          break
        case 'pattern2':
          this.createPatternBackground(ctx, '#43e97b', '#38f9d7')
          break
        case 'geometric':
          this.createGeometricBackground(ctx)
          break
        case 'dots':
          this.createDotsBackground(ctx)
          break
      }

      // 绘制拼图缺口
      this.drawPuzzleHole(ctx)
      this.bgImage = canvas.toDataURL()
    },

    createGradientBackground(ctx, colors) {
      const gradient = ctx.createLinearGradient(0, 0, 300, 150)
      gradient.addColorStop(0, colors[0])
      gradient.addColorStop(1, colors[1])
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, 300, 150)
    },

    createPatternBackground(ctx, color1, color2) {
      // 创建条纹背景
      const gradient = ctx.createLinearGradient(0, 0, 300, 150)
      gradient.addColorStop(0, color1)
      gradient.addColorStop(1, color2)
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, 300, 150)

      // 添加波浪效果
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
      ctx.lineWidth = 2
      for (let i = 0; i < 5; i++) {
        ctx.beginPath()
        ctx.moveTo(0, 30 * i)
        for (let x = 0; x < 300; x += 10) {
          ctx.lineTo(x, 30 * i + Math.sin(x * 0.02) * 10)
        }
        ctx.stroke()
      }
    },

    createGeometricBackground(ctx) {
      // 几何图形背景
      const gradient = ctx.createLinearGradient(0, 0, 300, 150)
      gradient.addColorStop(0, '#667eea')
      gradient.addColorStop(1, '#764ba2')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, 300, 150)

      // 添加几何图形
      ctx.fillStyle = 'rgba(255, 255, 255, 0.1)'
      for (let i = 0; i < 10; i++) {
        const x = Math.random() * 300
        const y = Math.random() * 150
        const size = Math.random() * 20 + 10

        if (Math.random() > 0.5) {
          // 圆形
          ctx.beginPath()
          ctx.arc(x, y, size, 0, 2 * Math.PI)
          ctx.fill()
        } else {
          // 矩形
          ctx.fillRect(x, y, size, size)
        }
      }
    },

    createDotsBackground(ctx) {
      // 点状背景
      const gradient = ctx.createRadialGradient(150, 75, 0, 150, 75, 200)
      gradient.addColorStop(0, '#a8edea')
      gradient.addColorStop(1, '#fed6e3')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, 300, 150)

      // 添加点状装饰
      for (let i = 0; i < 30; i++) {
        ctx.beginPath()
        ctx.arc(
          Math.random() * 300,
          Math.random() * 150,
          Math.random() * 3 + 1,
          0,
          2 * Math.PI
        )
        ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.8 + 0.2})`
        ctx.fill()
      }
    },

    drawPuzzleHole(ctx) {
      const x = this.puzzlePosition
      const y = 50
      const size = 50

      ctx.save()
      ctx.globalCompositeOperation = 'destination-out'

      switch(this.currentPuzzleType) {
        case 'square':
          ctx.fillRect(x, y, size, size)
          break
        case 'circle':
          ctx.beginPath()
          ctx.arc(x + size/2, y + size/2, size/2, 0, 2 * Math.PI)
          ctx.fill()
          break
        case 'triangle':
          ctx.beginPath()
          ctx.moveTo(x + size/2, y)
          ctx.lineTo(x, y + size)
          ctx.lineTo(x + size, y + size)
          ctx.closePath()
          ctx.fill()
          break
        case 'star':
          this.drawStar(ctx, x + size/2, y + size/2, 5, size/2, size/4)
          break
        case 'heart':
          this.drawHeart(ctx, x + size/2, y + size/2, size/2)
          break
        case 'diamond':
          ctx.beginPath()
          ctx.moveTo(x + size/2, y)
          ctx.lineTo(x + size, y + size/2)
          ctx.lineTo(x + size/2, y + size)
          ctx.lineTo(x, y + size/2)
          ctx.closePath()
          ctx.fill()
          break
      }

      ctx.restore()

      // 绘制缺口边框
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.shadowColor = 'rgba(0,0,0,0.5)'
      ctx.shadowBlur = 5
      ctx.stroke()
    },

    drawStar(ctx, x, y, spikes, outerRadius, innerRadius) {
      let rot = Math.PI / 2 * 3
      let step = Math.PI / spikes

      ctx.beginPath()
      ctx.moveTo(x, y - outerRadius)

      for (let i = 0; i < spikes; i++) {
        let x1 = x + Math.cos(rot) * outerRadius
        let y1 = y + Math.sin(rot) * outerRadius
        ctx.lineTo(x1, y1)
        rot += step

        let x2 = x + Math.cos(rot) * innerRadius
        let y2 = y + Math.sin(rot) * innerRadius
        ctx.lineTo(x2, y2)
        rot += step
      }

      ctx.lineTo(x, y - outerRadius)
      ctx.closePath()
      ctx.fill()
    },

    drawHeart(ctx, x, y, size) {
      ctx.beginPath()
      ctx.moveTo(x, y + size/4)
      ctx.bezierCurveTo(x, y, x - size/2, y, x - size/2, y + size/4)
      ctx.bezierCurveTo(x - size/2, y + size/2, x, y + size/2, x, y + size)
      ctx.bezierCurveTo(x, y + size/2, x + size/2, y + size/2, x + size/2, y + size/4)
      ctx.bezierCurveTo(x + size/2, y, x, y, x, y + size/4)
      ctx.closePath()
      ctx.fill()
    },

    getPuzzleMask() {
      const canvas = document.createElement('canvas')
      canvas.width = 50
      canvas.height = 50
      const ctx = canvas.getContext('2d')

      ctx.fillStyle = '#000'
      const size = 50

      switch(this.currentPuzzleType) {
        case 'square':
          ctx.fillRect(0, 0, size, size)
          break
        case 'circle':
          ctx.beginPath()
          ctx.arc(size/2, size/2, size/2, 0, 2 * Math.PI)
          ctx.fill()
          break
        case 'triangle':
          ctx.beginPath()
          ctx.moveTo(size/2, 0)
          ctx.lineTo(0, size)
          ctx.lineTo(size, size)
          ctx.closePath()
          ctx.fill()
          break
        case 'star':
          this.drawStar(ctx, size/2, size/2, 5, size/2, size/4)
          break
        case 'heart':
          this.drawHeart(ctx, size/2, size/2, size/2)
          break
        case 'diamond':
          ctx.beginPath()
          ctx.moveTo(size/2, 0)
          ctx.lineTo(size, size/2)
          ctx.lineTo(size/2, size)
          ctx.lineTo(0, size/2)
          ctx.closePath()
          ctx.fill()
          break
      }

      return canvas.toDataURL()
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
