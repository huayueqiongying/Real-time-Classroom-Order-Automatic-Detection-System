<template>
  <div class="camera-view">
    <div class="header">
      <h2>实时摄像头</h2>
      <div class="status-indicator">
        <div class="status-dot" :class="statusClass"></div>
        <span class="status-text">{{ statusText }}</span>
      </div>
    </div>

    <div class="video-container">
      <div class="video-wrapper">
        <img
          :src="videoFeedUrl"
          alt="实时视频流"
          @error="handleImageError"
          @load="handleImageLoad"
          class="video-stream"
          v-show="!isLoading && !hasError"
        />

        <!-- 加载状态 -->
        <div v-if="isLoading" class="overlay loading-overlay">
          <div class="loading-content">
            <div class="loading-spinner"></div>
            <p>正在连接摄像头...</p>
          </div>
        </div>

        <!-- 错误状态 -->
        <div v-if="hasError" class="overlay error-overlay">
          <div class="error-content">
            <div class="error-icon">⚠️</div>
            <h3>连接失败</h3>
            <p>{{ errorMessage }}</p>
            <button @click="retryConnection" class="retry-button">
              <span class="retry-icon">🔄</span>
              重新连接
            </button>
          </div>
        </div>
      </div>

      <!-- 控制栏 -->
      <div class="control-bar">
        <button @click="refreshStream" class="control-btn refresh-btn">
          <span class="btn-icon">🔄</span>
          刷新
        </button>
        <button @click="toggleFullscreen" class="control-btn fullscreen-btn">
          <span class="btn-icon">⛶</span>
          全屏
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CameraView',
  data() {
    return {
      videoFeedBaseUrl: 'http://127.0.0.1:5000/video_feed',
      streamId: '1', // 你的流ID是1
      isLoading: true,
      hasError: false,
      errorMessage: '',
      connectionStatus: 'connecting',
      retryCount: 0,
      maxRetries: 3
    };
  },
  computed: {
    videoFeedUrl() {
      return `${this.videoFeedBaseUrl}/${this.streamId}?t=${Date.now()}`;
    },
    statusClass() {
      return {
        'status-connected': this.connectionStatus === 'connected',
        'status-connecting': this.connectionStatus === 'connecting',
        'status-error': this.connectionStatus === 'error'
      };
    },
    statusText() {
      switch(this.connectionStatus) {
        case 'connected': return '在线';
        case 'connecting': return '连接中';
        case 'error': return '离线';
        default: return '未知';
      }
    }
  },
  methods: {
    handleImageLoad() {
      this.isLoading = false;
      this.hasError = false;
      this.connectionStatus = 'connected';
      this.retryCount = 0;
    },

    handleImageError() {
      this.isLoading = false;
      this.hasError = true;
      this.connectionStatus = 'error';
      this.errorMessage = '无法连接到摄像头，请检查设备状态';

      // 自动重试
      if (this.retryCount < this.maxRetries) {
        setTimeout(() => {
          this.retryConnection();
        }, 3000);
      }
    },

    retryConnection() {
      if (this.retryCount < this.maxRetries) {
        this.retryCount++;
        this.isLoading = true;
        this.hasError = false;
        this.connectionStatus = 'connecting';

        // 重新加载图片
        this.$nextTick(() => {
          const img = this.$el.querySelector('.video-stream');
          if (img) {
            img.src = this.videoFeedUrl;
          }
        });
      }
    },

    refreshStream() {
      this.isLoading = true;
      this.hasError = false;
      this.connectionStatus = 'connecting';
      this.retryCount = 0;

      this.$nextTick(() => {
        const img = this.$el.querySelector('.video-stream');
        if (img) {
          img.src = this.videoFeedUrl;
        }
      });
    },

    toggleFullscreen() {
      const videoWrapper = this.$el.querySelector('.video-wrapper');
      if (videoWrapper) {
        if (document.fullscreenElement) {
          document.exitFullscreen();
        } else {
          videoWrapper.requestFullscreen().catch(err => {
            console.error('无法进入全屏模式:', err);
          });
        }
      }
    }
  },

  mounted() {
    this.refreshStream();
  }
};
</script>

<style scoped>
.camera-view {
  padding: 24px;
  max-width: 1000px;
  margin: 0 auto;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 2px solid #f0f0f0;
}

.header h2 {
  margin: 0;
  font-size: 28px;
  font-weight: 600;
  color: #2c3e50;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.status-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.status-connected {
  background: #27ae60;
}

.status-connecting {
  background: #f39c12;
}

.status-error {
  background: #e74c3c;
}

.status-text {
  font-size: 14px;
  font-weight: 500;
  color: #34495e;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.video-container {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
}

.video-wrapper {
  position: relative;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  aspect-ratio: 16/9;
  margin-bottom: 16px;
}

.video-stream {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.85);
  backdrop-filter: blur(4px);
}

.loading-content, .error-content {
  text-align: center;
  color: white;
  padding: 20px;
}

.loading-spinner {
  width: 48px;
  height: 48px;
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-top: 4px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-content p {
  margin: 0;
  font-size: 16px;
  font-weight: 500;
}

.error-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.error-content h3 {
  margin: 0 0 8px 0;
  font-size: 20px;
  font-weight: 600;
}

.error-content p {
  margin: 0 0 20px 0;
  font-size: 14px;
  opacity: 0.9;
}

.retry-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  border: none;
  border-radius: 24px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s ease;
}

.retry-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
}

.retry-icon {
  font-size: 16px;
}

.control-bar {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.control-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  background: rgba(255, 255, 255, 0.95);
  border: none;
  border-radius: 24px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  color: #2c3e50;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.control-btn:hover {
  background: white;
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

.btn-icon {
  font-size: 16px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .camera-view {
    padding: 16px;
  }

  .header {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
  }

  .header h2 {
    font-size: 24px;
  }

  .video-container {
    padding: 12px;
  }

  .control-bar {
    flex-direction: column;
    align-items: stretch;
  }

  .control-btn {
    justify-content: center;
  }
}

/* 全屏模式样式 */
.video-wrapper:fullscreen {
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
}

.video-wrapper:fullscreen .video-stream {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
}
</style>
