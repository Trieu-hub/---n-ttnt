// web/js/faculty.js
(function () {
    // Ghi log khởi chạy
    console.log('faculty.js running');
    const params = new URLSearchParams(window.location.search);
    const fid = (params.get('id') || '').toUpperCase();
    console.log('Requested id:', fid);
    // Kiểm tra dữ liệu khoa
    if (typeof window.faculties === 'undefined') {
        console.warn('data.js not loaded or faculties undefined');
        window.faculties = {};
    }
    // Tìm khoa theo id
    const faculty = window.faculties[fid] || null;
    if (!faculty) {
        document.getElementById('faculty-name').textContent = 'Không tìm thấy khoa';
        document.getElementById('faculty-tagline').textContent = 'Vui lòng chọn khoa từ trang chính.';
        return;
    }
    // Cập nhật nội dung trang
    document.getElementById('faculty-name').textContent = faculty.name;
    document.getElementById('faculty-tagline').textContent = faculty.description;
    const subjUl = document.getElementById('subject-list');
    subjUl.innerHTML = '';
    (faculty.subjects || []).forEach(s => {
        const li = document.createElement('li');
        li.textContent = s;
        subjUl.appendChild(li);
    });

    // Ghi log hoàn tất
    console.log('Rendered faculty:', faculty.name);
})();
