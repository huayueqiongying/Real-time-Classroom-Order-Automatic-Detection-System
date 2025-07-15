<template>
  <div class="edit-profile-container">
    <div class="drop-zone" @drop.prevent="handleDrop" @dragover.prevent>
      <p>拖拽人脸图片到这里修改，或点击上传新照片</p>
      <input
        type="file"
        @change="handleFileSelect"
        accept="image/*"
        id="fileInput"
        style="display: none"
      >
      <label for="fileInput" class="upload-btn">选择文件</label>

      <div class="preview-area">
        <img
          :src="currentPhoto"
          class="preview-image"
          @error="handleImageError"
        />
      </div>
    </div>

    <div class="form-section">
      <h3>修改人员信息</h3>

      <div v-if="isLoading" class="loading-message">数据加载中...</div>
      <div v-if="loadError" class="error-message">{{ loadError }}</div>

      <div class="form-fields" v-if="!isLoading && !loadError">
        <div class="form-group">
          <label>姓名</label>
          <input
            type="text"
            v-model.trim="formData.name"
            @input="markFieldModified('name')"
            required
          >
          <span v-if="fieldModified.name" class="modified-indicator">*</span>
        </div>
        <div class="form-group">
          <label>性别</label>
          <select
            v-model="formData.gender"
            @change="markFieldModified('gender')"
          >
            <option value="male">男</option>
            <option value="female">女</option>
          </select>
          <span v-if="fieldModified.gender" class="modified-indicator">*</span>
        </div>
        <div class="form-group">
          <label>联系电话</label>
          <input
            type="tel"
            v-model.trim="formData.phone"
            @input="markFieldModified('phone')"
            required
          >
          <span v-if="fieldModified.phone" class="modified-indicator">*</span>
        </div>
        <div class="form-group">
          <label>{{ role === 'student' ? '学号' : '工号' }}</label>
          <input
            type="text"
            v-model="formData.id"
            readonly
            class="disabled-input"
          >
        </div>
      </div>

      <div class="action-buttons" v-if="!isLoading">
        <button @click="cancelEdit" class="cancel-btn">取消</button>
        <button
          @click="confirmSave"
          class="save-btn"
          :disabled="!isFormValid || !isFormModified || isSaving"
        >
          {{ isSaving ? '保存中...' : '保存' }}
        </button>
      </div>
    </div>

    <!-- 保存确认弹窗 -->
    <div v-if="showSaveModal" class="modal">
      <div class="modal-content">
        <p>是否保存修改后的信息？</p>
        <div class="modal-buttons">
          <button @click="saveChanges" class="confirm-btn">是</button>
          <button @click="showSaveModal = false" class="cancel-btn">否</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'EditProfile',
  props: {
    id: {
      type: String,
      required: true
    },
    role: {
      type: String,
      required: true,
      validator: value => ['student', 'teacher'].includes(value)
    }
  },
  data() {
    return {
      isLoading: false,
      isSaving: false,
      loadError: null,
      showSaveModal: false,
      selectedFile: null,
      currentPhoto: '',
      originalData: null,
      fieldModified: {
        name: false,
        gender: false,
        phone: false,
        photo: false
      },
      formData: {
        name: '',
        gender: 'male',
        phone: '',
        id: '',
        photo_path: ''
      }
    }
  },
  computed: {
    isFormValid() {
      return this.formData.name.trim() && this.formData.phone.trim()
    },
    isFormModified() {
      if (!this.originalData) return false
      return (
        this.formData.name !== this.originalData.name ||
        this.formData.gender !== this.originalData.gender ||
        this.formData.phone !== this.originalData.phone ||
        this.selectedFile !== null
      )
    }
  },
  created() {
    this.loadPersonData()
  },
  methods: {
    async loadPersonData() {
      this.isLoading = true
      this.loadError = null

      try {
        const response = await fetch(`http://localhost:3000/api/user-data/user/${this.id}`)
        if (!response.ok) throw new Error(`获取用户数据失败: ${response.status}`)

        const result = await response.json()
        if (!result.success) throw new Error(result.message || '获取用户数据失败')

        this.formData = {
          name: result.data.name,
          gender: result.data.gender,
          phone: result.data.phone,
          id: result.data.user_id,
          photo_path: result.data.photo_path || ''
        }

        this.originalData = JSON.parse(JSON.stringify(this.formData))

        // 加载照片
        await this.loadPhoto()
        this.resetModifiedFlags()
      } catch (error) {
        console.error('加载数据失败:', error)
        this.loadError = error.message
        this.formData.id = this.id
        this.currentPhoto = this.getDefaultPhoto()
      } finally {
        this.isLoading = false
      }
    },

    async loadPhoto() {
      try {
        // 1. 优先尝试直接加载学号/工号对应的图片
        const baseUrl = 'http://localhost:3000';
        const photoUrl = `${baseUrl}/uploads/${this.formData.id}.png?t=${Date.now()}`;

        // 直接创建Image对象检查图片是否存在
        const img = new Image();
        img.onload = () => {
          this.currentPhoto = photoUrl; // 图片存在则显示
        };
        img.onerror = () => {
          // 图片不存在则显示占位图
          this.currentPhoto = 'https://via.placeholder.com/200';
        };
        img.src = photoUrl;

      } catch (error) {
        console.error('照片加载失败:', error);
        this.currentPhoto = 'https://via.placeholder.com/200';
      }
    },

    async checkPhotoExists(url) {
      return new Promise((resolve) => {
        const img = new Image()
        img.onload = () => {
          this.currentPhoto = url
          resolve(true)
        }
        img.onerror = () => {
          resolve(false)
        }
        img.src = url
      })
    },

    getDefaultPhoto() {
      return 'https://via.placeholder.com/200'
    },

    handleImageError() {
      this.currentPhoto = this.getDefaultPhoto()
    },

    markFieldModified(field) {
      this.fieldModified[field] = true
    },

    resetModifiedFlags() {
      this.fieldModified = {
        name: false,
        gender: false,
        phone: false
      }
    },

    handleDrop(e) {
      const files = e.dataTransfer.files
      if (files.length > 0 && files[0].type.startsWith('image/')) {
        this.handleImageFile(files[0])
      }
    },

    handleFileSelect(e) {
      const files = e.target.files
      if (files.length > 0 && files[0].type.startsWith('image/')) {
        this.handleImageFile(files[0])
      }
    },

    handleImageFile(file) {
      this.selectedFile = file
      this.currentPhoto = URL.createObjectURL(file)
    },

    cancelEdit() {
      if (this.isFormModified && !confirm('确定要放弃未保存的修改吗？')) {
        return
      }
      this.$router.push('/students')
    },

    confirmSave() {
      if (!this.isFormValid) {
        alert('请填写所有必填字段')
        return
      }
      this.showSaveModal = true
    },

    async saveChanges() {
      this.isSaving = true
      this.showSaveModal = false

      try {
        // 1. 如果有新照片则先上传
        if (this.selectedFile) {
          const formData = new FormData()
          formData.append('image', this.selectedFile)
          formData.append('userId', this.formData.id)

          const uploadResponse = await fetch(`http://localhost:3000/api/user-data/user/${this.id}/photo`, {
            method: 'POST',
            body: formData
          })

          if (!uploadResponse.ok) {
            throw new Error('照片上传失败')
          }

          const uploadResult = await uploadResponse.json()
          this.formData.photo_path = uploadResult.path
        }

        // 2. 更新基本信息
        const updateResponse = await fetch(`http://localhost:3000/api/user-data/user/${this.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            name: this.formData.name,
            gender: this.formData.gender,
            phone: this.formData.phone,
            photo_path: this.formData.photo_path
          })
        })

        if (!updateResponse.ok) throw new Error('基本信息更新失败')

        const updateResult = await updateResponse.json()
        if (!updateResult.success) throw new Error(updateResult.message)

        // 3. 更新成功，返回列表页
        this.$router.push('/students')
      } catch (error) {
        console.error('保存失败:', error)
        alert(`保存失败: ${error.message}`)
      } finally {
        this.isSaving = false
      }
    }
  },
  beforeDestroy() {
    if (this.currentPhoto.startsWith('blob:')) {
      URL.revokeObjectURL(this.currentPhoto)
    }
  }
}
</script>

<style scoped>
/* 原有样式保持不变 */
.edit-profile-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Arial', sans-serif;
}

.drop-zone {
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 30px;
  text-align: center;
  margin-bottom: 30px;
  transition: all 0.3s;
}

.drop-zone:hover {
  border-color: #42b983;
  background-color: #f8f8f8;
}

.upload-btn {
  display: inline-block;
  padding: 10px 20px;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin: 15px 0;
  transition: background-color 0.3s;
}

.upload-btn:hover {
  background-color: #3aa876;
}

.preview-area {
  margin-top: 20px;
  position: relative;
  min-height: 200px;
}

.preview-image {
  max-width: 200px;
  max-height: 200px;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.form-section {
  background: #f9f9f9;
  padding: 25px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.loading-message,
.error-message {
  padding: 20px;
  text-align: center;
  margin: 20px 0;
  border-radius: 4px;
}

.loading-message {
  background-color: #f0f8ff;
  color: #0066cc;
}

.error-message {
  background-color: #ffebee;
  color: #c62828;
}

.form-fields {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
}

.form-group {
  margin-bottom: 20px;
  position: relative;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: #555;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  transition: border-color 0.3s;
}

.form-group input:focus,
.form-group select:focus {
  border-color: #42b983;
  outline: none;
}

.disabled-input {
  background-color: #f5f5f5;
  color: #666;
  cursor: not-allowed;
}

.modified-indicator {
  color: #f56c6c;
  margin-left: 5px;
  font-weight: bold;
}

.action-buttons {
  display: flex;
  justify-content: flex-end;
  gap: 15px;
  margin-top: 30px;
}

.cancel-btn {
  padding: 10px 20px;
  background-color: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.cancel-btn:hover {
  background-color: #e53935;
}

.save-btn {
  padding: 10px 20px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
  min-width: 80px;
}

.save-btn:hover:not(:disabled) {
  background-color: #43a047;
}

.save-btn:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
  opacity: 0.7;
}

.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0,0,0,0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  padding: 25px;
  border-radius: 8px;
  width: 350px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  text-align: center;
}

.modal-buttons {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 25px;
}

.confirm-btn {
  padding: 10px 20px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.confirm-btn:hover {
  background-color: #43a047;
}
</style>
