<template>
  <div class="danger-zone-view">
    <!-- 头部区域 -->
    <div class="header">
      <h2>危险区域检测</h2>
      <div class="header-controls">
        <div class="detection-status">
          <div class="status-dot" :class="detectionStatusClass"></div>
          <span class="status-text">{{ detectionStatusText }}</span>
        </div>
        <button @click="toggleDetection" class="toggle-btn" :class="{ active: isDetectionEnabled }">
          {{ isDetectionEnabled ? '停止检测' : '开始检测' }}
        </button>
      </div>
    </div>

    <!-- 主要内容区域 -->
    <div class="main-content">
      <!-- 左侧视频区域 -->
      <div class="video-section">
        <div class="video-container">
          <div class="video-wrapper" ref="videoWrapper"@click="handleVideoClick" @contextmenu="handleVideoRightClick">
            <img
              :src="videoFeedUrl"
              alt="危险区域检测视频流"
              @error="handleImageError"
              @load="handleImageLoad"
              class="video-stream"
              v-show="!isLoading && !hasError"
            />
            <!-- 绘制可视化层 -->
  <div v-if="isDrawing" class="drawing-overlay">
    <!-- 绘制的点标记 -->
    <div
      v-for="(point, index) in currentPolygon"
      :key="index"
      class="point-marker"
      :style="{
        left: point[0] + 'px',
        top: point[1] + 'px'
      }"
    >
      <span class="point-number">{{ index + 1 }}</span>
    </div>

    <!-- 绘制的线条 -->
    <svg class="polygon-preview" v-if="currentPolygon.length > 0">
      <!-- 已绘制的线条 -->
      <polyline
        v-if="currentPolygon.length > 1"
        :points="currentPolygon.map(p => p.join(',')).join(' ')"
        stroke="#3498db"
        stroke-width="2"
        fill="none"
        stroke-dasharray="5,5"
      />

      <!-- 闭合预览线（当有3个以上点时） -->
      <line
        v-if="currentPolygon.length > 2"
        :x1="currentPolygon[currentPolygon.length - 1][0]"
        :y1="currentPolygon[currentPolygon.length - 1][1]"
        :x2="currentPolygon[0][0]"
        :y2="currentPolygon[0][1]"
        stroke="#3498db"
        stroke-width="2"
        stroke-dasharray="10,5"
        opacity="0.6"
      />

      <!-- 填充预览（当有3个以上点时） -->
      <polygon
        v-if="currentPolygon.length > 2"
        :points="currentPolygon.map(p => p.join(',')).join(' ')"
        fill="rgba(52, 152, 219, 0.2)"
        stroke="rgba(52, 152, 219, 0.5)"
        stroke-width="1"
      />
    </svg>
  </div>

  <!-- 已配置区域的可视化显示 -->
  <div v-if="!isDrawing && config.zones.length > 0" class="zones-overlay">
    <svg class="zones-preview">
      <g v-for="(zone, index) in config.zones" :key="index">
        <polygon
          :points="zone.polygon.map(p => p.join(',')).join(' ')"
          fill="rgba(231, 76, 60, 0.3)"
          stroke="rgba(231, 76, 60, 0.8)"
          stroke-width="2"
        />
        <text
          :x="getZoneCenter(zone.polygon)[0]"
          :y="getZoneCenter(zone.polygon)[1]"
          fill="white"
          font-size="14"
          font-weight="bold"
          text-anchor="middle"
          dominant-baseline="middle"
        >
          {{ zone.name }}
        </text>
      </g>
    </svg>
  </div>


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

          <!-- 视频控制栏 -->
          <div class="video-controls">
            <button @click="refreshStream" class="control-btn">
              <span class="btn-icon">🔄</span>
              刷新
            </button>
            <button @click="toggleFullscreen" class="control-btn">
              <span class="btn-icon">⛶</span>
              全屏
            </button>
          </div>
        </div>
      </div>

      <!-- 右侧控制面板 -->
      <div class="control-panel">
        <!-- 区域配置 -->
        <div class="panel-section">
          <h3>区域配置</h3>
          <div class="config-form">
            <div class="form-group">
              <label>安全距离 (像素):</label>
              <input
                type="number"
                v-model="config.safety_distance"
                min="10"
                max="200"
                class="form-input"
                @change="updateConfig"
              />
            </div>
            <div class="form-group">
              <label>停留时间 (秒):</label>
              <input
                type="number"
                v-model="config.stay_time"
                min="1"
                max="60"
                class="form-input"
                @change="updateConfig"
              />
            </div>
            <div class="form-group">
              <label>区域名称:</label>
              <input
                type="text"
                v-model="newZoneName"
                placeholder="输入区域名称"
                class="form-input"
              />
            </div>
            <button @click="startDrawing" class="action-btn" :disabled="isDrawing">
              {{ isDrawing ? '绘制中...' : '绘制新区域' }}
            </button>
          </div>
        </div>

        <!-- 已配置区域列表 -->
        <div class="panel-section">
          <h3>已配置区域</h3>
          <div class="zones-list">
            <div v-if="config.zones.length === 0" class="empty-message">
              暂无配置的危险区域
            </div>
            <div
              v-for="(zone, index) in config.zones"
              :key="index"
              class="zone-item"
            >
              <div class="zone-info">
                <span class="zone-name">{{ zone.name }}</span>
                <span class="zone-points">{{ zone.polygon.length }} 个点</span>
              </div>
              <div class="zone-actions">
                <button @click="deleteZone(index)" class="delete-btn">删除</button>
              </div>
            </div>
          </div>
        </div>

        <!-- 告警信息 -->
        <div class="panel-section">
          <h3>告警信息</h3>
          <div class="alerts-container">
            <div class="alerts-header">
              <span class="alerts-count">共 {{ alerts.length }} 条告警</span>
              <button @click="clearAlerts" class="clear-btn">清除告警</button>
            </div>
            <div class="alerts-list" style="max-height: 300px; min-height: 150px;">
              <div v-if="alerts.length === 0" class="empty-message">
                暂无告警信息
              </div>
              <div
                v-for="alert in alerts.slice(0, 10)"
                :key="alert.timestamp"
                class="alert-item"
                :class="{ 'alert-high': alert.severity === 'high' }"
              >
                <div class="alert-content">
                  <div class="alert-title">
                    {{ alert.zone_name }}
                  </div>
                  <div class="alert-details">
                    {{ getAlertText(alert) }}
                  </div>
                  <div class="alert-time">
                    {{ formatTime(alert.timestamp) }}
                  </div>
                </div>
                <div class="alert-severity" :class="alert.severity">
                  {{ alert.severity === 'high' ? '高' : '中' }}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 绘制提示 -->
    <div v-if="isDrawing" class="draw-instructions">
      <div class="instructions-content">
        <h4>绘制危险区域</h4>
        <p>点击视频画面绘制多边形区域，右键完成绘制</p>
        <div class="instructions-actions">
          <button @click="cancelDrawing" class="cancel-btn">取消</button>
          <button @click="completeDrawing" class="complete-btn" :disabled="currentPolygon.length < 3">
            完成绘制
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DangerZoneView',
  data() {
    return {
      baseUrl: 'http://127.0.0.1:5000',
      streamId: '1',
      isLoading: true,
      hasError: false,
      errorMessage: '',
      connectionStatus: 'connecting',
      retryCount: 0,
      maxRetries: 3,

      // 检测配置
      config: {
        enabled: false,
        zones: [],
        safety_distance: 50,
        stay_time: 3
      },

      // 告警信息
      alerts: [],

      // 添加视频尺寸数据
      videoWidth: 640,
      videoHeight: 480,

      // 绘制相关
      isDrawing: false,
      currentPolygon: [],
      newZoneName: '',

      // 定时器
      alertTimer: null,
      configTimer: null
    };
  },

  computed: {
    videoFeedUrl() {
      return `${this.baseUrl}/danger_feed/${this.streamId}?t=${Date.now()}`;
    },

    detectionStatusClass() {
      return {
        'status-enabled': this.config.enabled,
        'status-disabled': !this.config.enabled
      };
    },

    detectionStatusText() {
      return this.config.enabled ? '检测中' : '已停止';
    },

    isDetectionEnabled() {
      return this.config.enabled;
    }
  },

  methods: {
    // 加载配置
    async loadConfig() {
      try {
        const response = await fetch(`${this.baseUrl}/danger_zones/${this.streamId}`);
        const data = await response.json();
        this.config = data;
      } catch (error) {
        console.error('加载配置失败:', error);
      }
    },


    // 保存配置
    async saveConfig() {
      try {
        const response = await fetch(`${this.baseUrl}/danger_zones/${this.streamId}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(this.config)
        });
        const data = await response.json();
        if (response.ok) {
          this.$message.success(data.message || '配置保存成功');
        } else {
          this.$message.error(data.error || '配置保存失败');
        }
      } catch (error) {
        console.error('保存配置失败:', error);
        this.$message.error('配置保存失败');
      }
    },

    // 更新配置
    updateConfig() {
      clearTimeout(this.configTimer);
      this.configTimer = setTimeout(() => {
        this.saveConfig();
      }, 1000);
    },

    // 切换检测状态
    async toggleDetection() {
      try {
        const response = await fetch(`${this.baseUrl}/danger_zones/${this.streamId}/toggle`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ enabled: !this.config.enabled })
        });
        const data = await response.json();
        if (response.ok) {
          this.config.enabled = !this.config.enabled;
          this.$message.success(data.message || '状态更新成功');
        } else {
          this.$message.error(data.error || '状态更新失败');
        }
      } catch (error) {
        console.error('切换检测状态失败:', error);
        this.$message.error('状态更新失败');
      }
    },

    // 开始绘制
    startDrawing() {
      if (!this.newZoneName.trim()) {
        this.$message.warning('请输入区域名称');
        return;
      }

      this.isDrawing = true;
      this.currentPolygon = [];
      this.bindDrawingEvents();
    },

    // 绑定绘制事件
    bindDrawingEvents() {
      this.$nextTick(() => {
        const videoElement = this.$el.querySelector('.video-stream');
        if (videoElement) {
          videoElement.addEventListener('click', this.handleVideoClick);
          videoElement.addEventListener('contextmenu', this.handleVideoRightClick);
        }
      });
    },

    // 解绑绘制事件
    unbindDrawingEvents() {
      const videoElement = this.$el.querySelector('.video-stream');
      if (videoElement) {
        videoElement.removeEventListener('click', this.handleVideoClick);
        videoElement.removeEventListener('contextmenu', this.handleVideoRightClick);
      }
    },
    // 计算区域中心点（用于显示区域名称）
    getZoneCenter(polygon) {
      if (polygon.length === 0) return [0, 0];

      const sumX = polygon.reduce((sum, point) => sum + point[0], 0);
      const sumY = polygon.reduce((sum, point) => sum + point[1], 0);

      return [sumX / polygon.length, sumY / polygon.length];
    },

    // 修改处理视频点击方法，添加视觉反馈
    handleVideoClick(event) {
      if (!this.isDrawing) return;


      // 获取视频元素的实际尺寸和位置
      const videoElement = this.$el.querySelector('.video-stream');
      if (!videoElement) return;

      const rect = event.target.getBoundingClientRect();
      const x = Math.round(event.clientX - rect.left);
      const y = Math.round(event.clientY - rect.top);
      // 确保坐标在视频范围内
      if (x >= 0 && x < rect.width && y >= 0 && y < rect.height) {
        this.currentPolygon.push([x, y]);
        console.log('添加点:', [x, y]);
      }

      // 添加点击反馈
      this.showClickFeedback(x, y);
    },
    // 添加点击反馈效果
    showClickFeedback(x, y) {
      const feedback = document.createElement('div');
      feedback.className = 'click-feedback';
      feedback.style.left = x + 'px';
      feedback.style.top = y + 'px';

      const videoWrapper = this.$refs.videoWrapper;
      videoWrapper.appendChild(feedback);

      // 动画结束后移除元素
      setTimeout(() => {
        if (feedback.parentNode) {
          feedback.parentNode.removeChild(feedback);
        }
      }, 600);
    },

    // 处理右键点击
    handleVideoRightClick(event) {
      if (!this.isDrawing) return;

      event.preventDefault();
      this.completeDrawing();
    },

    // 完成绘制
    completeDrawing() {
      if (this.currentPolygon.length < 3) {
        this.$message.warning('至少需要3个点才能形成区域');
        return;
      }

      const newZone = {
        name: this.newZoneName.trim(),
        polygon: [...this.currentPolygon]
      };

      this.config.zones.push(newZone);
      this.saveConfig();

      this.cancelDrawing();
      this.newZoneName = '';
      this.$message.success('区域添加成功');
    },

    // 取消绘制
    cancelDrawing() {
      this.isDrawing = false;
      this.currentPolygon = [];
      this.unbindDrawingEvents();
    },

    // 删除区域
    deleteZone(index) {
      if (confirm('确定要删除此区域吗？')) {
        this.config.zones.splice(index, 1);
        this.saveConfig();
        this.$message.success('区域删除成功');
      }
    },

    // 加载告警
    async loadAlerts() {
      try {
        console.log('正在加载告警...');
        const response = await fetch(`${this.baseUrl}/danger_alerts/${this.streamId}`);
        const data = await response.json();
        console.log('告警数据:', data);
        this.alerts = data.alerts || [];
        console.log('当前告警数量:', this.alerts.length);
      } catch (error) {
        console.error('加载告警失败:', error);
      }
    },

    // 清除告警
    async clearAlerts() {
      try {
        const response = await fetch(`${this.baseUrl}/danger_alerts/clear/${this.streamId}`, {
          method: 'POST'
        });
        const data = await response.json();
        if (response.ok) {
          this.alerts = [];
          this.$message.success(data.message || '告警清除成功');
        } else {
          this.$message.error(data.error || '告警清除失败');
        }
      } catch (error) {
        console.error('清除告警失败:', error);
        this.$message.error('告警清除失败');
      }
    },

    // 获取告警文本
    getAlertText(alert) {
      if (alert.alert_type === 'intrusion') {
        return `人员闯入危险区域 (${alert.person_id})`;
      } else if (alert.alert_type === 'proximity') {
        return `人员距离危险区域过近 (${alert.person_id})，距离: ${Math.round(alert.distance)}px`;
      }
      return `未知告警类型`;
    },

    // 格式化时间
    formatTime(timestamp) {
      const date = new Date(timestamp);
      return date.toLocaleString('zh-CN');
    },

    // 视频加载成功
    handleImageLoad() {
      this.isLoading = false;
      this.hasError = false;
      this.connectionStatus = 'connected';
      this.retryCount = 0;
      // 获取视频实际尺寸
      this.$nextTick(() => {
        const videoElement = this.$el.querySelector('.video-stream');
        if (videoElement) {
          this.videoWidth = videoElement.clientWidth;
          this.videoHeight = videoElement.clientHeight;
        }
      });
    },

    // 视频加载失败
    handleImageError() {
      this.isLoading = false;
      this.hasError = true;
      this.connectionStatus = 'error';
      this.errorMessage = '无法连接到摄像头，请检查设备状态';

      if (this.retryCount < this.maxRetries) {
        setTimeout(() => {
          this.retryConnection();
        }, 3000);
      }
    },

    // 重试连接
    retryConnection() {
      if (this.retryCount < this.maxRetries) {
        this.retryCount++;
        this.isLoading = true;
        this.hasError = false;
        this.connectionStatus = 'connecting';

        this.$nextTick(() => {
          const img = this.$el.querySelector('.video-stream');
          if (img) {
            img.src = this.videoFeedUrl;
          }
        });
      }
    },

    // 刷新视频流
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

    // 切换全屏
    toggleFullscreen() {
      const videoWrapper = this.$refs.videoWrapper;
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
    this.loadConfig();
    this.loadAlerts();
    this.refreshStream();

    // 定时刷新告警
    this.alertTimer = setInterval(() => {
      this.loadAlerts();
    }, 2000);
  },

  beforeDestroy() {
    // 清理定时器
    if (this.alertTimer) {
      clearInterval(this.alertTimer);
    }
    if (this.configTimer) {
      clearTimeout(this.configTimer);
    }

    // 解绑事件
    this.unbindDrawingEvents();
  }
};
</script>

<style scoped>
.danger-zone-view {
  padding: 20px;
  background: #f5f5f5;
  min-height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 20px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.header h2 {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
  color: #2c3e50;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 20px;
}

.detection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: #f8f9fa;
  border-radius: 20px;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.status-enabled {
  background: #27ae60;
}

.status-disabled {
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

.toggle-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s ease;
  background: #e74c3c;
  color: white;
}

.toggle-btn.active {
  background: #27ae60;
}

.toggle-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.main-content {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;
  height: calc(100vh - 120px);
}

.video-section {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.video-container {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.video-wrapper {
  position: relative;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  flex: 1;
  margin-bottom: 16px;
}

.video-stream {
  width: 100%;
  height: 100%;
  object-fit: cover;
  cursor: crosshair;
}

.video-stream:not(.drawing) {
  cursor: default;
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
  background: rgba(0, 0, 0, 0.8);
}

.loading-content, .error-content {
  text-align: center;
  color: white;
  padding: 20px;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top: 3px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-icon {
  font-size: 40px;
  margin-bottom: 16px;
}

.retry-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s ease;
  margin: 0 auto;
}

.retry-button:hover {
  background: #2980b9;
}

.video-controls {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.control-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s ease;
}

.control-btn:hover {
  background: #e9ecef;
}

.control-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
  overflow-y: auto; /* 修改为auto以显示滚动条 */
  max-height: 100%; /* 保证不会超出父容器 */
}

.panel-section {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.panel-section h3 {
  margin: 0 0 16px 0;
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
}

.config-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-group label {
  font-size: 14px;
  font-weight: 500;
  color: #34495e;
}

.form-input {
  padding: 10px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  transition: border-color 0.3s ease;
}

.form-input:focus {
  outline: none;
  border-color: #3498db;
}

.action-btn {
  padding: 12px 20px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s ease;
}

.action-btn:hover:not(:disabled) {
  background: #2980b9;
}

.action-btn:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}

.zones-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 200px;
  overflow-y: auto;
}

.zone-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #dee2e6;
}

.zone-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.zone-name {
  font-size: 14px;
  font-weight: 500;
  color: #2c3e50;
}

.zone-points {
  font-size: 12px;
  color: #7f8c8d;
}

.delete-btn {
  padding: 6px 12px;
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.3s ease;
}

.delete-btn:hover {
  background: #c0392b;
}

.alerts-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
}

.alerts-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.alerts-count {
  font-size: 14px;
  color: #7f8c8d;
}

.clear-btn {
  padding: 6px 12px;
  background: #95a5a6;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.3s ease;
}

.clear-btn:hover {
  background: #7f8c8d;
}

.alerts-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.alert-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 6px;
  border-left: 4px solid #f39c12;
}

.alert-item.alert-high {
  background: #f8d7da;
  border-color: #f5c6cb;
  border-left-color: #e74c3c;
}

.alert-content {
  flex: 1;
}

.alert-title {
  font-size: 14px;
  font-weight: 500;
  color: #2c3e50;
  margin-bottom: 4px;
}

.alert-details {
  font-size: 12px;
  color: #7f8c8d;
  margin-bottom: 4px;
}

.alert-time {
  font-size: 11px;
  color: #bdc3c7;
}

.alert-severity {
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  color: white;
}

.alert-severity.medium {
  background: #f39c12;
}

.alert-severity.high {
  background: #e74c3c;
}

.empty-message {
  text-align: center;
  color: #7f8c8d;
  font-size: 14px;
  padding: 20px;
}

.draw-instructions {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  z-index: 1000;
}

.instructions-content {
  text-align: center;
}

.instructions-content h4 {
  margin: 0 0 8px 0;
  font-size: 16px;
  color: #2c3e50;
}

.instructions-content p {
  margin: 0 0 16px 0;
  font-size: 14px;
  color: #7f8c8d;
}

.instructions-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.cancel-btn {
  padding: 8px 16px;
  background: #95a5a6;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s ease;
}

.cancel-btn:hover {
  background: #7f8c8d;
}

.complete-btn {
  padding: 8px 16px;
  background: #27ae60;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s ease;
}

.complete-btn:hover:not(:disabled) {
  background: #219653;
}

.complete-btn:disabled {
  background: #95a5a6;
  cursor: not-allowed;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .main-content {
    grid-template-columns: 1fr;
    height: auto;
  }

  .video-section {
    height: 70vh;
  }
}

@media (max-width: 768px) {
  .header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }

  .header-controls {
    width: 100%;
    justify-content: space-between;
  }

  .video-controls {
    flex-wrap: wrap;
  }

  .control-btn {
    flex: 1;
    min-width: 120px;
    margin-bottom: 8px;
  }

  .panel-section {
    padding: 15px;
  }

  .form-group {
    flex-direction: column;
    align-items: flex-start;
  }

  .draw-instructions {
    width: 90%;
    padding: 16px;
  }
}

@media (max-width: 480px) {
  .danger-zone-view {
    padding: 10px;
  }

  .header h2 {
    font-size: 20px;
  }

  .toggle-btn {
    padding: 8px 15px;
    font-size: 13px;
  }

  .video-controls {
    flex-direction: column;
  }

  .control-btn {
    width: 100%;
  }

  .zone-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }

  .zone-actions {
    align-self: flex-end;
  }

  .alert-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .alert-severity {
    align-self: flex-end;
  }
}

/* 绘制点标记样式 */
.point-marker {
  position: absolute;
  width: 12px;
  height: 12px;
  background: #3498db;
  border: 2px solid white;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  z-index: 10;
}

.point-line {
  position: absolute;
  background: #3498db;
  height: 2px;
  z-index: 9;
  transform-origin: 0 0;
}

/* 绘制区域预览 */
.polygon-preview {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 8;
}
.video-stream.drawing {
  cursor: crosshair;
}

.drawing-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 10;
}

.zones-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 5;
}

.polygon-preview,
.zones-preview {
  width: 100%;
  height: 100%;
}

.point-marker {
  position: absolute;
  width: 20px;
  height: 20px;
  background: #3498db;
  border: 3px solid white;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  z-index: 15;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.point-number {
  color: white;
  font-size: 10px;
  font-weight: bold;
}

.click-feedback {
  position: absolute;
  width: 30px;
  height: 30px;
  border: 2px solid #3498db;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  animation: clickPulse 0.6s ease-out;
  pointer-events: none;
  z-index: 20;
}

@keyframes clickPulse {
  0% {
    transform: translate(-50%, -50%) scale(0.5);
    opacity: 1;
  }
  100% {
    transform: translate(-50%, -50%) scale(2);
    opacity: 0;
  }
}

/* 优化绘制提示样式 */
.draw-instructions {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.instructions-content h4 {
  margin: 0 0 8px 0;
  font-size: 16px;
  color: #2c3e50;
  text-align: center;
}

.instructions-content p {
  margin: 0 0 16px 0;
  font-size: 14px;
  color: #7f8c8d;
  text-align: center;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .point-marker {
    width: 16px;
    height: 16px;
  }

  .point-number {
    font-size: 8px;
  }

  .click-feedback {
    width: 25px;
    height: 25px;
  }
}
</style>
