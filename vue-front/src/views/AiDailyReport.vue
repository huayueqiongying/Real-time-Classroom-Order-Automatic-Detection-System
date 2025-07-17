<template>
  <div class="daily-reports">
    <div class="header">
      <h2>监控日报</h2>
      <div class="header-actions">
        <button @click="showDatePicker" class="generate-btn secondary">
          <i class="icon-calendar">📅</i>
          指定日期生成
        </button>
        <button @click="generateReport" :disabled="generating" class="generate-btn">
          <i class="icon-generate">📊</i>
          {{ generating ? '正在生成...' : '生成昨日报告' }}
        </button>
      </div>
    </div>

    <div v-if="generating" class="generating-indicator">
      <div class="spinner"></div>
      <p>AI正在生成报告中，请稍候...</p>
    </div>

    <div v-if="!generating && reports.length === 0" class="no-reports">
      <p>暂无监控日报</p>
      <div class="no-reports-actions">
        <button @click="generateReport" class="btn-generate">生成昨日报告</button>
        <button @click="showDatePicker" class="btn-generate secondary">指定日期生成</button>
      </div>
    </div>

    <div v-if="!generating && reports.length > 0" class="reports-list">
      <div v-for="report in reports" :key="report.id" class="report-card">
        <div class="report-header">
          <h3>{{ report.date }} 监控日报</h3>
          <span class="report-time">{{ formatTime(report.created_at) }}</span>
        </div>

        <div class="report-content">
          <pre>{{ report.content }}</pre>
        </div>

        <div class="report-actions">
          <button @click="downloadReport(report)" class="btn-download">
            <i class="icon-download">📥</i> 下载文本
          </button>
          <button @click="confirmDelete(report)" class="btn-delete">
            <i class="icon-delete">🗑️</i> 删除
          </button>
        </div>
      </div>
    </div>

    <div v-if="totalPages > 1" class="pagination">
      <button @click="changePage(currentPage - 1)" :disabled="currentPage === 1">
        上一页
      </button>
      <span>第 {{ currentPage }} 页 / 共 {{ totalPages }} 页</span>
      <button @click="changePage(currentPage + 1)" :disabled="currentPage === totalPages">
        下一页
      </button>
    </div>

    <!-- 日期选择器弹窗 -->
    <div v-if="datePickerDialog.show" class="modal-overlay" @click="closeDatePicker">
      <div class="modal-content date-picker-modal" @click.stop>
        <h3>选择日期生成报告</h3>

        <div class="date-picker-content">
          <div class="form-group">
            <label for="report-date">选择日期:</label>
            <input
              type="date"
              id="report-date"
              v-model="datePickerDialog.selectedDate"
              :min="availableDateRange.min_date"
              :max="availableDateRange.max_date"
              @change="checkDateData"
              class="date-input"
            >
          </div>

          <div v-if="datePickerDialog.selectedDate" class="date-info">
            <div v-if="datePickerDialog.loading" class="loading-info">
              <div class="mini-spinner"></div>
              <span>检查数据中...</span>
            </div>

            <div v-else-if="datePickerDialog.dateInfo" class="date-status">
              <div class="info-item">
                <i class="icon-info">ℹ️</i>
                <span>该日期有 {{ datePickerDialog.dateInfo.event_count }} 条监控数据</span>
              </div>

              <div v-if="datePickerDialog.dateInfo.report_exists" class="info-item warning">
                <i class="icon-warning">⚠️</i>
                <span>该日期的报告已存在</span>
              </div>

              <div v-if="!datePickerDialog.dateInfo.has_data" class="info-item warning">
                <i class="icon-warning">⚠️</i>
                <span>该日期没有监控数据</span>
              </div>
            </div>
          </div>

          <div v-if="datePickerDialog.dateInfo && datePickerDialog.dateInfo.report_exists" class="force-generate">
            <label class="checkbox-label">
              <input
                type="checkbox"
                v-model="datePickerDialog.forceGenerate"
                class="checkbox-input"
              >
              <span class="checkbox-text">强制重新生成（会覆盖现有报告）</span>
            </label>
          </div>
        </div>

        <div class="modal-actions">
          <button @click="closeDatePicker" class="btn-cancel">取消</button>
          <button
            @click="generateReportForDate"
            :disabled="!canGenerateReport"
            class="btn-confirm"
          >
            {{ datePickerDialog.generating ? '生成中...' : '生成报告' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 删除确认弹窗 -->
    <div v-if="deleteDialog.show" class="modal-overlay" @click="cancelDelete">
      <div class="modal-content" @click.stop>
        <h3>确认删除</h3>
        <p>确定要删除 "{{ deleteDialog.report.date }}" 的监控日报吗？</p>
        <div class="modal-actions">
          <button @click="cancelDelete" class="btn-cancel">取消</button>
          <button @click="deleteReport" class="btn-confirm">确认删除</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'DailyReports',
  data() {
    return {
      reports: [],
      generating: false,
      currentPage: 1,
      totalPages: 1,
      perPage: 5,
      availableDateRange: {
        min_date: null,
        max_date: null,
        total_events: 0
      },
      datePickerDialog: {
        show: false,
        selectedDate: '',
        dateInfo: null,
        loading: false,
        forceGenerate: false,
        generating: false
      },
      deleteDialog: {
        show: false,
        report: null
      }
    }
  },
  computed: {
    canGenerateReport() {
      if (!this.datePickerDialog.selectedDate || this.datePickerDialog.generating) {
        return false;
      }

      const dateInfo = this.datePickerDialog.dateInfo;
      if (!dateInfo) {
        return false;
      }

      // 如果没有数据，不能生成报告
      if (!dateInfo.has_data) {
        return false;
      }

      // 如果已有报告且没有选择强制生成，不能生成
      if (dateInfo.report_exists && !this.datePickerDialog.forceGenerate) {
        return false;
      }

      return true;
    }
  },
  mounted() {
    this.loadReports();
    this.loadAvailableDateRange();
  },
  methods: {
    async loadReports() {
      try {
        const response = await axios.get('http://127.0.0.1:5000/daily_reports', {
          params: {
            page: this.currentPage,
            per_page: this.perPage
          }
        });

        this.reports = response.data.reports;
        this.totalPages = Math.ceil(response.data.total / this.perPage);
      } catch (error) {
        console.error('加载报告失败:', error);
        this.showErrorMessage('加载报告失败');
      }
    },

    async loadAvailableDateRange() {
      try {
        const response = await axios.get('http://127.0.0.1:5000/available_dates');
        this.availableDateRange = response.data;
      } catch (error) {
        console.error('加载可用日期范围失败:', error);
      }
    },

    async generateReport() {
      this.generating = true;
      try {
        const response = await axios.post('http://127.0.0.1:5000/generate_daily_report');

        // 添加新报告到列表顶部
        const newReport = {
          id: response.data.id,
          date: response.data.date,
          content: response.data.content,
          created_at: response.data.created_at
        };

        this.reports.unshift(newReport);
        this.showSuccessMessage('日报生成成功！');

      } catch (error) {
        console.error('生成报告失败:', error);

        // 处理重复生成的情况
        if (error.response && error.response.status === 409) {
          this.showErrorMessage('当日报告已存在，请勿重复生成');
        } else {
          const errorMsg = error.response && error.response.data && error.response.data.error || '服务器错误';
          this.showErrorMessage('日报生成失败: ' + errorMsg);
        }
      } finally {
        this.generating = false;
      }
    },

    showDatePicker() {
      this.datePickerDialog.show = true;
      this.datePickerDialog.selectedDate = '';
      this.datePickerDialog.dateInfo = null;
      this.datePickerDialog.forceGenerate = false;
    },

    closeDatePicker() {
      this.datePickerDialog.show = false;
      this.datePickerDialog.selectedDate = '';
      this.datePickerDialog.dateInfo = null;
      this.datePickerDialog.forceGenerate = false;
      this.datePickerDialog.generating = false;
    },

    async checkDateData() {
      if (!this.datePickerDialog.selectedDate) {
        this.datePickerDialog.dateInfo = null;
        return;
      }

      this.datePickerDialog.loading = true;
      try {
        const response = await axios.post('http://127.0.0.1:5000/check_date_data', {
          date: this.datePickerDialog.selectedDate
        });

        this.datePickerDialog.dateInfo = response.data;
        this.datePickerDialog.forceGenerate = false;
      } catch (error) {
        console.error('检查日期数据失败:', error);
        this.showErrorMessage('检查日期数据失败');
      } finally {
        this.datePickerDialog.loading = false;
      }
    },

    async generateReportForDate() {
      if (!this.canGenerateReport) {
        return;
      }

      this.datePickerDialog.generating = true;
      try {
        const requestData = {
          date: this.datePickerDialog.selectedDate
        };

        // 如果需要强制生成，添加 force 标记
        if (this.datePickerDialog.forceGenerate) {
          requestData.force = true;
        }

        const response = await axios.post('http://127.0.0.1:5000/generate_daily_report', requestData);

        // 添加新报告到列表顶部
        const newReport = {
          id: response.data.id,
          date: response.data.date,
          content: response.data.content,
          created_at: response.data.created_at
        };

        // 如果是强制生成，先移除可能存在的旧报告
        if (this.datePickerDialog.forceGenerate) {
          this.reports = this.reports.filter(report => report.date !== newReport.date);
        }

        this.reports.unshift(newReport);
        this.showSuccessMessage(`${newReport.date} 日报生成成功！`);
        this.closeDatePicker();

      } catch (error) {
        console.error('生成指定日期报告失败:', error);
        const errorMsg = error.response && error.response.data && error.response.data.error || '服务器错误';
        this.showErrorMessage('生成报告失败: ' + errorMsg);
      } finally {
        this.datePickerDialog.generating = false;
      }
    },

    confirmDelete(report) {
      this.deleteDialog.show = true;
      this.deleteDialog.report = report;
    },

    cancelDelete() {
      this.deleteDialog.show = false;
      this.deleteDialog.report = null;
    },

    async deleteReport() {
      try {
        const reportId = this.deleteDialog.report.id;
        await axios.delete(`http://127.0.0.1:5000/daily_reports/${reportId}`);

        // 从列表中移除删除的报告
        this.reports = this.reports.filter(report => report.id !== reportId);

        this.showSuccessMessage('报告删除成功！');
        this.cancelDelete();

        // 如果当前页没有报告了，跳转到上一页
        if (this.reports.length === 0 && this.currentPage > 1) {
          this.currentPage--;
          this.loadReports();
        }

      } catch (error) {
        console.error('删除报告失败:', error);
        const errorMsg = error.response && error.response.data && error.response.data.error || '服务器错误';
        this.showErrorMessage('删除报告失败: ' + errorMsg);
        this.cancelDelete();
      }
    },

    downloadReport(report) {
      const blob = new Blob([report.content], { type: 'text/plain' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `监控日报_${report.date}.txt`;
      link.click();
      URL.revokeObjectURL(link.href);
    },

    formatTime(timeString) {
      return new Date(timeString).toLocaleString('zh-CN');
    },

    changePage(page) {
      if (page >= 1 && page <= this.totalPages) {
        this.currentPage = page;
        this.loadReports();
      }
    },

    showSuccessMessage(message) {
      // 如果使用了 Element UI 或其他消息组件
      if (this.$message) {
        this.$message.success(message);
      } else {
        alert(message);
      }
    },

    showErrorMessage(message) {
      // 如果使用了 Element UI 或其他消息组件
      if (this.$message) {
        this.$message.error(message);
      } else {
        alert(message);
      }
    }
  }
}
</script>

<style scoped>
.daily-reports {
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.generate-btn {
  padding: 10px 15px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  transition: background 0.3s;
}

.generate-btn.secondary {
  background: #95a5a6;
}

.generate-btn:hover:not(:disabled) {
  background: #2980b9;
}

.generate-btn.secondary:hover:not(:disabled) {
  background: #7f8c8d;
}

.generate-btn:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}

.generating-indicator {
  text-align: center;
  padding: 30px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 8px;
  margin: 20px 0;
}

.spinner {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3498db;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 0 auto 15px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.no-reports {
  text-align: center;
  padding: 40px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.no-reports-actions {
  display: flex;
  gap: 15px;
  justify-content: center;
  margin-top: 20px;
}

.btn-generate {
  padding: 10px 20px;
  background: #2ecc71;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.btn-generate.secondary {
  background: #95a5a6;
}

.btn-generate:hover {
  background: #27ae60;
}

.btn-generate.secondary:hover {
  background: #7f8c8d;
}

.reports-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.report-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  overflow: hidden;
}

.report-header {
  padding: 15px 20px;
  background: #2c3e50;
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.report-header h3 {
  margin: 0;
  font-size: 1.2rem;
}

.report-time {
  font-size: 0.9rem;
  opacity: 0.8;
}

.report-content {
  padding: 20px;
  max-height: 300px;
  overflow-y: auto;
  background: #fcfcfc;
  border-bottom: 1px solid #eee;
}

.report-content pre {
  white-space: pre-wrap;
  font-family: 'Microsoft YaHei', sans-serif;
  line-height: 1.6;
  margin: 0;
}

.report-actions {
  padding: 15px 20px;
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.btn-download {
  padding: 8px 15px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
}

.btn-download:hover {
  background: #2980b9;
}

.btn-delete {
  padding: 8px 15px;
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
}

.btn-delete:hover {
  background: #c0392b;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 15px;
  margin-top: 20px;
  padding: 10px;
}

.pagination button {
  padding: 8px 15px;
  background: #ecf0f1;
  border: 1px solid #bdc3c7;
  border-radius: 4px;
  cursor: pointer;
}

.pagination button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 删除确认弹窗样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  padding: 20px;
  border-radius: 8px;
  max-width: 400px;
  width: 90%;
  text-align: center;
}

.modal-content h3 {
  margin-top: 0;
  color: #2c3e50;
}

.modal-content p {
  margin: 15px 0;
  color: #7f8c8d;
}

.modal-actions {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 20px;
}

.btn-cancel {
  padding: 8px 20px;
  background: #95a5a6;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.btn-cancel:hover {
  background: #7f8c8d;
}

.btn-confirm {
  padding: 8px 20px;
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.btn-confirm:hover {
  background: #c0392b;
}
</style>
