<template>
  <div class="event-handling">
    <div class="header">
      <h2>异常事件处理</h2>
      <div class="stats">
        <div class="stat-item">
          <span class="stat-number">{{ totalEvents }}</span>
          <span class="stat-label">总事件</span>
        </div>
        <div class="stat-item">
          <span class="stat-number">{{ pendingEvents }}</span>
          <span class="stat-label">待处理</span>
        </div>
        <div class="stat-item">
          <span class="stat-number">{{ handledEvents }}</span>
          <span class="stat-label">已处理</span>
        </div>
      </div>
    </div>

    <!-- 筛选和搜索 -->
    <div class="filters">
      <div class="filter-group">
        <label>状态筛选：</label>
        <select v-model="selectedStatus" @change="loadEvents">
          <option value="all">全部</option>
          <option value="pending">待处理</option>
          <option value="handled">已处理</option>
        </select>
      </div>

      <div class="filter-group">
        <label>事件类型：</label>
        <select v-model="selectedEventType" @change="loadEvents">
          <option value="all">全部</option>
          <option value="bad_behavior">异常行为</option>
          <option value="stranger">陌生人</option>
        </select>
      </div>

      <div class="filter-group">
        <button @click="loadEvents" class="refresh-btn">
          <i class="icon-refresh">🔄</i>
          刷新
        </button>
      </div>
    </div>

    <!-- 事件列表 -->
    <div class="events-container">
      <div v-if="loading" class="loading">
        <div class="spinner"></div>
        <p>加载中...</p>
      </div>

      <div v-else-if="events.length === 0" class="no-events">
        <p>暂无异常事件</p>
      </div>

      <div v-else class="events-grid">
        <div
          v-for="event in events"
          :key="event.id"
          class="event-card"
          :class="{ 'handled': event.status === 'handled' }"
        >
          <div class="event-header">
            <div class="event-type">
              <span
                class="type-badge"
                :class="event.event_type"
              >
                {{ getEventTypeLabel(event.event_type) }}
              </span>
              <span class="behavior-class">{{ event.behavior_class }}</span>
            </div>
            <div class="event-time">
              {{ formatTime(event.created_at) }}
            </div>
          </div>

          <div class="event-details">
            <div class="detail-item">
              <span class="label">流ID：</span>
              <span class="value">{{ event.stream_id }}</span>
            </div>
            <div class="detail-item">
              <span class="label">用户：</span>
              <span class="value" :class="{ 'stranger': event.student_id === 'Stranger' }">
                {{ event.student_id === 'Stranger' ? '陌生人' : event.student_id }}
              </span>
            </div>
            <div class="detail-item">
              <span class="label">置信度：</span>
              <span class="value confidence">{{ (event.confidence * 100).toFixed(1) }}%</span>
            </div>
          </div>

          <div class="event-actions">
            <button
              @click="downloadVideo(event)"
              class="btn btn-primary"
              :disabled="!event.video_path"
            >
              <i class="icon-download">📥</i>
              下载视频
            </button>

            <button
              v-if="event.status === 'pending'"
              @click="handleEvent(event)"
              class="btn btn-success"
            >
              <i class="icon-check">✓</i>
              标记已处理
            </button>

            <button
              @click="showEventDetails(event)"
              class="btn btn-info"
            >
              <i class="icon-info">ℹ️</i>
              详情
            </button>
          </div>

          <div v-if="event.status === 'handled'" class="handled-info">
            <p>已于 {{ formatTime(event.handled_at) }} 由 {{ event.handler }} 处理</p>
          </div>
        </div>
      </div>
    </div>

    <!-- 分页 -->
    <div v-if="totalPages > 1" class="pagination">
      <button
        @click="changePage(currentPage - 1)"
        :disabled="currentPage === 1"
        class="btn btn-secondary"
      >
        上一页
      </button>

      <span class="page-info">
        第 {{ currentPage }} 页 / 共 {{ totalPages }} 页
      </span>

      <button
        @click="changePage(currentPage + 1)"
        :disabled="currentPage === totalPages"
        class="btn btn-secondary"
      >
        下一页
      </button>
    </div>

    <!-- 事件详情模态框 -->
    <div v-if="detailModal.show" class="modal-overlay" @click="closeDetailModal">
      <div class="modal-content detail-modal" @click.stop>
        <div class="modal-header">
          <h3>事件详情</h3>
          <button @click="closeDetailModal" class="close-btn">&times;</button>
        </div>
        <div class="modal-body">
          <div class="detail-grid">
            <div class="detail-row">
              <span class="detail-label">事件ID：</span>
              <span class="detail-value">{{ detailModal.event.id }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">流ID：</span>
              <span class="detail-value">{{ detailModal.event.stream_id }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">事件类型：</span>
              <span class="detail-value">{{ getEventTypeLabel(detailModal.event.event_type) }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">行为分类：</span>
              <span class="detail-value">{{ detailModal.event.behavior_class }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">置信度：</span>
              <span class="detail-value">{{ (detailModal.event.confidence * 100).toFixed(2) }}%</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">涉及用户：</span>
              <span class="detail-value">{{ detailModal.event.student_id === 'Stranger' ? '陌生人' : detailModal.event.student_id }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">检测时间：</span>
              <span class="detail-value">{{ formatTime(detailModal.event.created_at) }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">状态：</span>
              <span class="detail-value">
                <span :class="['status-badge', detailModal.event.status]">
                  {{ detailModal.event.status === 'pending' ? '待处理' : '已处理' }}
                </span>
              </span>
            </div>
            <div v-if="detailModal.event.status === 'handled'" class="detail-row">
              <span class="detail-label">处理时间：</span>
              <span class="detail-value">{{ formatTime(detailModal.event.handled_at) }}</span>
            </div>
            <div v-if="detailModal.event.status === 'handled'" class="detail-row">
              <span class="detail-label">处理人：</span>
              <span class="detail-value">{{ detailModal.event.handler }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'EventHandling',
  data() {
    return {
      events: [],
      loading: false,
      totalEvents: 0,
      pendingEvents: 0,
      handledEvents: 0,
      selectedStatus: 'all',
      selectedEventType: 'all',
      currentPage: 1,
      totalPages: 1,
      perPage: 12,

      detailModal: {
        show: false,
        event: null
      },

      // 自动刷新
      refreshInterval: null
    }
  },

  async mounted() {
    await this.loadEvents()
    this.startAutoRefresh()
  },

  beforeDestroy() {
    this.stopAutoRefresh()
  },

  methods: {
    async loadEvents() {
      this.loading = true
      try {
        const response = await axios.get('http://127.0.0.1:5000/anomaly_events', {
          params: {
            page: this.currentPage,
            per_page: this.perPage,
            status: this.selectedStatus
          }
        })

        this.events = response.data.events
        this.totalPages = response.data.total_pages
        this.updateStats()

      } catch (error) {
        console.error('加载事件失败:', error)
        // 修复：移除可选链操作符
        if (this.$message && this.$message.error) {
          this.$message.error('加载事件失败')
        }
      } finally {
        this.loading = false
      }
    },

    async updateStats() {
      try {
        // 获取所有状态的统计
        const [allResponse, pendingResponse, handledResponse] = await Promise.all([
          axios.get('http://127.0.0.1:5000/anomaly_events', { params: { status: 'all', per_page: 1 } }),
          axios.get('http://127.0.0.1:5000/anomaly_events', { params: { status: 'pending', per_page: 1 } }),
          axios.get('http://127.0.0.1:5000/anomaly_events', { params: { status: 'handled', per_page: 1 } })
        ])

        this.totalEvents = allResponse.data.total
        this.pendingEvents = pendingResponse.data.total
        this.handledEvents = handledResponse.data.total

      } catch (error) {
        console.error('更新统计失败:', error)
      }
    },

    async changePage(page) {
      if (page >= 1 && page <= this.totalPages) {
        this.currentPage = page
        await this.loadEvents()
      }
    },

    async handleEvent(event) {
      try {
        const handler = prompt('请输入处理人姓名:')
        if (!handler) return

        await axios.post(`http://127.0.0.1:5000/anomaly_events/${event.id}/handle`, {
          handler: handler
        })

        // 重新加载事件列表
        await this.loadEvents()
        alert('事件处理成功')

      } catch (error) {
        console.error('处理事件失败:', error)
        alert('处理事件失败')
      }
    },

    // 修改为下载视频的方法
    downloadVideo(event) {
      try {
        // 构造下载URL
        const downloadUrl = `http://127.0.0.1:5000/anomaly_events/${event.id}/video`

        // 方法1：创建隐藏的a标签进行下载
        const link = document.createElement('a')
        link.href = downloadUrl
        link.download = `event_${event.id}_video.mp4` // 设置下载文件名
        link.style.display = 'none'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)

        // 方法2：如果方法1不工作，直接打开新窗口
        // window.open(downloadUrl, '_blank')

        console.log('开始下载视频:', downloadUrl)

      } catch (error) {
        console.error('下载视频失败:', error)
        alert('下载视频失败: ' + error.message)

        // 备用方案：直接在新窗口打开
        window.open(`http://127.0.0.1:5000/anomaly_events/${event.id}/video`, '_blank')
      }
    },

    showEventDetails(event) {
      this.detailModal.event = event
      this.detailModal.show = true
    },

    closeDetailModal() {
      this.detailModal.show = false
      this.detailModal.event = null
    },

    getEventTypeLabel(type) {
      const labels = {
        'bad_behavior': '异常行为',
        'stranger': '陌生人'
      }
      return labels[type] || type
    },

    formatTime(timeString) {
      if (!timeString) return '--'
      const date = new Date(timeString)
      return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      })
    },

    startAutoRefresh() {
      // 每30秒自动刷新一次
      this.refreshInterval = setInterval(() => {
        this.loadEvents()
      }, 30000)
    },

    stopAutoRefresh() {
      if (this.refreshInterval) {
        clearInterval(this.refreshInterval)
        this.refreshInterval = null
      }
    }
  }
}
</script>

<style scoped>
.event-handling {
  padding: 20px;
  background: #f5f5f5;
  min-height: 100vh;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header h2 {
  margin: 0;
  color: #333;
}

.stats {
  display: flex;
  gap: 30px;
}

.stat-item {
  text-align: center;
}

.stat-number {
  display: block;
  font-size: 24px;
  font-weight: bold;
  color: #2c3e50;
}

.stat-label {
  font-size: 14px;
  color: #7f8c8d;
}

.filters {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  background: white;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.filter-group label {
  font-weight: 500;
  color: #333;
}

.filter-group select {
  padding: 5px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.refresh-btn {
  padding: 8px 16px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
}

.refresh-btn:hover {
  background: #2980b9;
}

.events-container {
  min-height: 400px;
}

.loading {
  text-align: center;
  padding: 50px;
}

.spinner {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3498db;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 2s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.no-events {
  text-align: center;
  padding: 50px;
  color: #7f8c8d;
}

.events-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 20px;
}

.event-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: transform 0.2s;
}

.event-card:hover {
  transform: translateY(-2px);
}

.event-card.handled {
  border-left: 4px solid #27ae60;
}

.event-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 15px;
}

.event-type {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.type-badge {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
  color: white;
  text-align: center;
}

.type-badge.bad_behavior {
  background: #e74c3c;
}

.type-badge.stranger {
  background: #f39c12;
}

.behavior-class {
  font-weight: 500;
  color: #2c3e50;
}

.event-time {
  font-size: 12px;
  color: #7f8c8d;
}

.event-details {
  margin-bottom: 15px;
}

.detail-item {
  display: flex;
  margin-bottom: 8px;
}

.detail-item .label {
  font-weight: 500;
  color: #555;
  min-width: 60px;
}

.detail-item .value {
  color: #333;
}

.detail-item .value.stranger {
  color: #f39c12;
  font-weight: 500;
}

.detail-item .value.confidence {
  color: #27ae60;
  font-weight: 500;
}

.event-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.btn {
  padding: 8px 12px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 5px;
  transition: background 0.2s;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary {
  background: #3498db;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #2980b9;
}

.btn-success {
  background: #27ae60;
  color: white;
}

.btn-success:hover {
  background: #219a52;
}

.btn-info {
  background: #95a5a6;
  color: white;
}

.btn-info:hover {
  background: #7f8c8d;
}

.btn-secondary {
  background: #bdc3c7;
  color: #2c3e50;
}

.btn-secondary:hover:not(:disabled) {
  background: #95a5a6;
}

.handled-info {
  margin-top: 10px;
  padding: 8px;
  background: #d5edda;
  border-radius: 4px;
  font-size: 12px;
  color: #155724;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 20px;
  margin-top: 30px;
}

.page-info {
  color: #7f8c8d;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 8px;
  max-width: 90vw;
  max-height: 90vh;
  overflow: auto;
}

.detail-modal {
  width: 600px;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #eee;
}

.modal-header h3 {
  margin: 0;
  color: #333;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #999;
}

.close-btn:hover {
  color: #333;
}

.modal-body {
  padding: 20px;
}

.detail-grid {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.detail-row {
  display: flex;
  padding: 10px;
  border-bottom: 1px solid #f0f0f0;
}

.detail-label {
  font-weight: 500;
  color: #555;
  min-width: 100px;
}

.detail-value {
  color: #333;
  flex: 1;
}

.status-badge {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
  color: white;
}

.status-badge.pending {
  background: #f39c12;
}

.status-badge.handled {
  background: #27ae60;
}
</style>
