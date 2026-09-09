document.getElementById('processBtn').addEventListener('click', async () => {
  const text = document.getElementById('textInput').value.trim();
  const funcType = document.getElementById('funcSelect').value;
  const resultBox = document.getElementById('resultBox');
  const resultContent = document.getElementById('resultContent');

  if (!text) {
    alert('请先输入文本！');
    return;
  }

  // 显示加载状态
  resultBox.style.display = 'block';
  resultContent.innerHTML = '<p style="color: #666;">处理中...</p>';

  try {
    // 发送POST请求到Django后端
    const response = await fetch('/nlp/process/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-CSRFToken': csrfToken  // 带上CSRF Token
      },
      body: `text=${encodeURIComponent(text)}&func_type=${funcType}`
    });

    const data = await response.json();

    if (data.status === 'success') {
      resultContent.innerHTML = `<p style="font-size: 16px; color: #333;">${data.result}</p>`;
    } else {
      resultContent.innerHTML = `<p style="color: #dc2626;">${data.msg}</p>`;
    }
  } catch (error) {
    resultContent.innerHTML = `<p style="color: #dc2626;">请求失败：${error.message}</p>`;
    console.error(error);
  }
});