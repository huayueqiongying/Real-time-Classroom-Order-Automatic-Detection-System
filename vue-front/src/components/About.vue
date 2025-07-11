<template>
  <div class="about">
    <h2>系统简介</h2>
<p>「智能教室监控系统」是基于Vue.js+Django的现代化教育管理平台，主要功能包括：</p>

<h3>核心技术</h3>
<ul>
  <li><strong>多摄像头融合</strong>：支持摄像头实时接入，低延迟直播</li>
  <li><strong>AI行为分析</strong>：通过AI识别举手、离座、使用手机等行为</li>
  <li><strong>物联网集成</strong>：联动教室监控等教室设备</li>
</ul>

<h3>应用场景</h3>
<ul>
  <li>🔒 安防监控：陌生人检测、危险区域预警</li>
  <li>🎓 教学评估：学生专注度分析、教师板书质量检测</li>
  <li>📅 资源管理：学生信息存储、设备维护提醒</li>
</ul>

<p style="font-style: italic;">"通过数字化手段，让每一间教室都成为可感知、可交互、可优化的智能空间"</p>
  </div>
</template>

<script>
export default {
  name: 'About'
}
</script>

<style scoped>
.about {
  margin: 20px auto;
  max-width: 500px;
  padding: 20px;
  border: 1px solid #eee;
  border-radius: 5px;
}
</style>
