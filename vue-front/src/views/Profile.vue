<template>
  <!-- 模板部分保持不变 -->
  <div class="profile-container">
    <div
      class="drop-zone"
      @drop.prevent="handleDrop"
      @dragover.prevent
    >
      <p>拖拽人脸图片到这里，或点击上传</p>
      <input
        type="file"
        @change="handleFileSelect"
        accept="image/*"
        id="fileInput"
        style="display: none"
      >
      <label for="fileInput" class="upload-btn">选择文件</label>

      <div v-if="previewUrl" class="preview-area">
        <img :src="previewUrl" class="preview-image" />
        <p v-if="!selectedFile">未选择任何文件</p>
      </div>
    </div>

    <div class="form-section">
      <h3>人员信息</h3>
      <div class="role-selector">
        <label>
          <input type="radio" v-model="role" value="teacher"> 老师
        </label>
        <label>
          <input type="radio" v-model="role" value="student"> 学生
        </label>
      </div>

      <div class="form-fields">
        <!-- 公共字段 -->
        <div class="form-group">
          <label>姓名</label>
          <input type="text" v-model="formData.name" required>
        </div>
        <div class="form-group">
          <label>性别</label>
          <select v-model="formData.gender">
            <option value="male">男</option>
            <option value="female">女</option>
          </select>
        </div>
        <div class="form-group">
          <label>联系电话</label>
          <input type="tel" v-model="formData.phone" required>
        </div>

        <!-- 老师专属字段 -->
        <div class="form-group" v-if="role === 'teacher'">
          <label>工号 <small>(格式：T开头+数字，如T001)</small></label>
          <input
            type="text"
            v-model="formData.teacherId"
            pattern="^T\d+$"
            required
          >
        </div>

        <!-- 学生专属字段 -->
        <div class="form-group" v-if="role === 'student'">
          <label>学号 <small>(格式：S开头+数字，如S2023001)</small></label>
          <input
            type="text"
            v-model="formData.studentId"
            pattern="^S\d+$"
            required
          >
        </div>
      </div>

      <button
        @click="submit"
        class="submit-btn"
        :disabled="!canSubmit"
      >
        提交信息
      </button>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      selectedFile: null,
      previewUrl: "",
      role: "teacher", // 默认老师
      formData: {
        name: "",
        gender: "male",
        phone: "",
        teacherId: "",
        studentId: ""
      }
    };
  },
  computed: {
    canSubmit() {
      if (!this.selectedFile) return false;
      if (!this.formData.name.trim() || !this.formData.phone.trim()) return false;

      if (this.role === 'teacher') {
        return /^T\d+$/.test(this.formData.teacherId);
      } else {
        return /^S\d+$/.test(this.formData.studentId);
      }
    },
    currentUserId() {
      return this.role === 'teacher'
        ? this.formData.teacherId
        : this.formData.studentId;
    }
  },
  methods: {
    handleDrop(e) {
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].type.startsWith('image/')) {
        this.selectedFile = files[0];
        this.previewImage();
      }
    },
    handleFileSelect(e) {
      const files = e.target.files;
      if (files.length > 0 && files[0].type.startsWith('image/')) {
        this.selectedFile = files[0];
        this.previewImage();
      }
    },
    previewImage() {
      if (!this.selectedFile) return;
      if (this.previewUrl) {
        URL.revokeObjectURL(this.previewUrl);
      }
      this.previewUrl = URL.createObjectURL(this.selectedFile);
    },
    // 在methods中添加新的上传照片方法
    async uploadPhoto() {
  if (!this.selectedFile || !this.currentUserId) {
    throw new Error('缺少图片或用户ID');
  }

  try {
    // 创建FormData对象
    const formData = new FormData();
    formData.append('image', this.selectedFile);
    formData.append('userId', this.currentUserId);

    const response = await axios.post('http://localhost:3000/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    if (response.data.success) {
      console.log('照片上传成功:', response.data);
      return response.data;
    } else {
      throw new Error(response.data.message || '照片上传失败');
    }
  } catch (error) {
    console.error('照片上传失败:', error);
    throw error;
  }
},

    async saveUserToDatabase() {
      try {
        await axios.post('http://localhost:3000/api/save-user-data', {
          userId: this.currentUserId,
          name: this.formData.name,
          gender: this.formData.gender,
          phone: this.formData.phone,
          role: this.role
        });
      } catch (error) {
        console.error('数据库保存失败:', error);
        throw error; // 抛出错误由调用者处理
      }
    },
    // 将图片转换为base64格式
  async convertFileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        // 去掉data:image/...;base64,前缀，只保留base64数据
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  },

  // 调用后端的人脸注册功能
  async registerFace() {
    if (!this.selectedFile || !this.currentUserId) {
      throw new Error('缺少图片或用户ID');
    }

    try {
      const imageBase64 = await this.convertFileToBase64(this.selectedFile);

      const response = await axios.post('http://localhost:5000/face-register', {
        student_id: this.currentUserId, // 使用当前用户ID（老师或学生）
        image: imageBase64
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (response.data.message === 'Face registered successfully') {
        console.log('人脸注册成功:', response.data);
        return response.data;
      } else {
        throw new Error(response.data.error || '人脸注册失败');
      }
    } catch (error) {
      console.error('人脸注册失败:', error);
      throw error;
    }
  },

    // 修改原有的submit方法
  async submit() {
    if (!this.canSubmit || this.isSubmitting) return;

    this.isSubmitting = true;

    try {
      // 1. 先进行人脸注册
      await this.registerFace();
      // 2. 保存用户信息到数据库 (你现有的逻辑)
      await this.saveUserToDatabase();

      alert(`提交成功！\n用户ID: ${this.currentUserId}\n人脸特征已注册到系统`);
      this.resetForm();

    } catch (error) {
      console.error('提交失败:', error);
      let errorMsg = '提交失败';

      if (error.response && error.response.data) {
        errorMsg = error.response.data.error || error.response.data.message || errorMsg;
      } else if (error.message) {
        errorMsg = error.message;
      }

      alert('错误: ' + errorMsg);
    } finally {
      this.isSubmitting = false;
    }
  },
    resetForm() {
      this.selectedFile = null;
      if (this.previewUrl) {
        URL.revokeObjectURL(this.previewUrl);
      }
      this.previewUrl = "";
      this.formData = {
        name: "",
        gender: "male",
        phone: "",
        teacherId: "",
        studentId: ""
      };
    }






  },
  beforeDestroy() {
    if (this.previewUrl) {
      URL.revokeObjectURL(this.previewUrl);
    }
  }
};
</script>

<style>
/* 样式部分保持不变 */
.profile-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.drop-zone {
  border: 2px dashed #ccc;
  padding: 20px;
  text-align: center;
  margin-bottom: 30px;
  transition: border-color 0.3s;
}

.drop-zone:hover {
  border-color: #42b983;
}

.upload-btn {
  display: inline-block;
  padding: 10px 20px;
  background-color: #42b983;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  margin: 10px 0;
  border: none;
  transition: background-color 0.3s;
}

.upload-btn:hover {
  background-color: #369f6e;
}

.preview-image {
  max-width: 100%;
  max-height: 400px;
  margin: 15px 0;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.form-section {
  background: #f9f9f9;
  padding: 25px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.role-selector {
  margin: 15px 0;
  display: flex;
  gap: 20px;
}

.role-selector label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.form-fields {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.form-group {
  display: flex;
  flex-direction: column;
  margin-bottom: 15px;
}

.form-group label {
  margin-bottom: 8px;
  font-weight: 500;
  color: #333;
}

.form-group small {
  color: #666;
  font-size: 0.8em;
}

.form-group input,
.form-group select {
  padding: 10px 12px;
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

.submit-btn {
  background-color: #42b983;
  color: white;
  padding: 12px 24px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-top: 20px;
  font-size: 16px;
  transition: all 0.3s;
  width: 100%;
}

.submit-btn:hover {
  background-color: #369f6e;
}

.submit-btn:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}
</style>
