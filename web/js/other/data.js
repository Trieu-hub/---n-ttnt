// Data about faculties
window.faculties = {
    IT: {
        name: "Khoa Công nghệ thông tin",
        description: "Đào tạo về lập trình, AI, mạng máy tính...",
        overview: "Khoa CNTT cung cấp nhiều chương trình đào tạo từ cơ bản đến nâng cao...",
        research: {
        labs: ["Phòng lab AI", "Phòng lab IoT", "Phòng lab An ninh mạng"],
        },
        resources: [
        { label: "MIT OCW", url: "https://ocw.mit.edu" },
        { label: "Coursera", url: "https://coursera.org" }
        ],
        career_prospects: "Phát triển phần mềm, Quản trị mạng, Chuyên gia an ninh mạng...",
        email: "cntt@hutech.edu.vn",
        phone: "028 3512 0789",
        website: "https://hutech.edu.vn"
    },

    PHARMA: {
        name: "Khoa Dược",
        description: "Đào tạo dược sĩ, nghiên cứu dược phẩm và chăm sóc sức khỏe.",
        overview: "Khoa Dược cung cấp kiến thức về bào chế, kiểm nghiệm và quản lý thuốc.",
        research: {
        labs: ["Phòng thí nghiệm Hóa dược", "Phòng thí nghiệm Dược lý"],

        },
        resources: [
        { label: "PubMed", url: "https://pubmed.ncbi.nlm.nih.gov" }
        ],
        career_prospects: "Dược sĩ, Nghiên cứu viên, Chuyên gia kiểm nghiệm.",
        email: "pharma@hutech.edu.vn",
        phone: "028 3512 0790",
        website: "https://hutech.edu.vn"
    },

    SSHPR: {
        name: "Khoa Khoa học Xã hội & Nhân văn",
        description: "Đào tạo các chuyên ngành xã hội học, quan hệ công chúng, tâm lý học.",
        overview: "Khoa SSHPR giúp sinh viên hiểu sâu về xã hội, truyền thông và văn hóa.",
        research: {
        labs: ["Phòng nghiên cứu xã hội học", "Phòng nghiên cứu truyền thông"],
        },
        resources: [
        { label: "Google Scholar", url: "https://scholar.google.com" }
        ],
                career_prospects: "Nhà nghiên cứu xã hội, Chuyên viên truyền thông, Tư vấn tâm lý.",
        email: "sshpr@hutech.edu.vn",
        phone: "028 3512 0791",
        website: "https://hutech.edu.vn"
    },

    LAW: {
        name: "Khoa Luật",
        description: "Đào tạo cử nhân luật, nghiên cứu pháp lý và tư pháp.",
        overview: "Sinh viên sẽ được trang bị kiến thức pháp luật trong nhiều lĩnh vực.",
        research: {
        labs: ["Trung tâm nghiên cứu pháp luật"],
        },
        resources: [
        { label: "Legal Information Institute", url: "https://www.law.cornell.edu" }
        ],
                career_prospects: "Luật sư, Chuyên viên pháp lý, Giảng viên luật.",
        email: "law@hutech.edu.vn",
        phone: "028 3512 0792",
        website: "https://hutech.edu.vn"
    },

    JAPAN: {
        name: "Khoa Nhật Bản học",
        description: "Đào tạo ngôn ngữ, văn hóa và quan hệ Nhật Bản.",
        overview: "Khoa cung cấp các chương trình về tiếng Nhật và văn hóa Nhật.",
        research: {
        labs: ["Trung tâm nghiên cứu Nhật Bản"],
        },
        resources: [
        { label: "NHK World Japan", url: "https://www3.nhk.or.jp/nhkworld" }
        ],
        career_prospects: "Phiên dịch viên, Giảng viên tiếng Nhật, Chuyên gia văn hóa Nhật Bản.",
        email: "japan@hutech.edu.vn",
        phone: "028 3512 0793",
        website: "https://hutech.edu.vn"
    },

    FINANCE: {
        name: "Khoa Tài chính – Ngân hàng",
        description: "Đào tạo chuyên sâu về tài chính, đầu tư, ngân hàng.",
        overview: "Sinh viên được trang bị kiến thức quản lý tài chính và phân tích đầu tư.",
        research: {
        labs: ["Phòng thí nghiệm mô phỏng thị trường tài chính"],
        },
        resources: [
        { label: "Investopedia", url: "https://www.investopedia.com" }
        ],
        career_prospects: "Chuyên viên phân tích tài chính, Quản lý rủi ro, Cố vấn đầu tư.",
        email: "finance@hutech.edu.vn",
        phone: "028 3512 0794",
        website: "https://hutech.edu.vn"
    },

    BA: {
        name: "Khoa Quản trị Kinh doanh",
        description: "Đào tạo kiến thức và kỹ năng quản trị doanh nghiệp.",
        overview: "Sinh viên được học các kỹ năng quản lý, khởi nghiệp, và phát triển doanh nghiệp.",
        research: {
        labs: ["Phòng nghiên cứu chiến lược kinh doanh"]
        },
        resources: [
        { label: "Harvard Business Review", url: "https://hbr.org" }
        ],
                career_prospects: "Chuyên viên quản trị, Chuyên viên marketing, Quản lý dự án.",
        email: "ba@hutech.edu.vn",
        phone: "028 3512 0795",
        website: "https://hutech.edu.vn"
    },

    MD: {
        name: "Khoa Media-Design",
        description: "Đào tạo chuyên sâu về thiết kế truyền thông, sáng tạo đa phương tiện và ứng dụng công nghệ trong thiết kế.",
        overview: "Khoa Media-Design cung cấp kiến thức và kỹ năng thực hành trong các lĩnh vực thiết kế đồ họa, thiết kế tương tác, và truyền thông đa phương tiện.",
        research: {
            labs: ["Phòng thí nghiệm Thiết kế Đa phương tiện", "Trung tâm Nghiên cứu Sáng tạo Truyền thông"]
        },
        resources: [
            { label: "Adobe Creative Cloud", url: "https://www.adobe.com/creativecloud.html" }
        ],
        career_prospects: "Nhà thiết kế đồ họa, Chuyên gia UX/UI, Quản lý dự án truyền thông, Chuyên viên sáng tạo đa phương tiện.",
        email: "md@hutech.edu.vn",
        phone: "028 3512 0796",
        website: "https://hutech.edu.vn"
    },

    VKIT: {
        name: "Viện Công nghệ Việt - Hàn",
        description: "Đào tạo liên kết Việt Nam – Hàn Quốc về công nghệ.",
        overview: "Viện VKIT hợp tác đào tạo kỹ sư công nghệ với các trường đại học Hàn Quốc.",
        research: {
        labs: ["Phòng nghiên cứu hợp tác Việt - Hàn"]
        },
        resources: [
        { label: "KOSEN Japan", url: "https://www.kosen-k.go.jp" }
        ],
        career_prospects: "Kỹ sư công nghệ, Chuyên gia robot, Nhà nghiên cứu trí tuệ nhân tạo.",
        email: "vkit@hutech.edu.vn",
        phone: "028 3512 0797",
        website: "https://hutech.edu.vn"
    },

    IEI: {
        name: "Viện Đào tạo Quốc tế",
        description: "Đào tạo quốc tế, liên kết với các trường đại học nước ngoài.",
        overview: "Sinh viên có cơ hội học tập trong môi trường quốc tế, chương trình trao đổi.",
        research: {
        labs: ["Trung tâm hợp tác quốc tế"]
        },
        resources: [
        { label: "Study Abroad", url: "https://studyabroad.com" }
        ],
        career_prospects: "Giảng viên, Nghiên cứu viên, Chuyên gia tư vấn giáo dục.",
        email: "iei@hutech.edu.vn",
        phone: "028 3512 0798",
        website: "https://hutech.edu.vn"
    },

    KOREAN: {
        name: "Khoa Hàn Quốc học",
        description: "Đào tạo ngôn ngữ, văn hóa và kinh tế Hàn Quốc.",
        overview: "Khoa cung cấp kiến thức về tiếng Hàn và văn hóa Hàn Quốc.",
        research: {
        labs: ["Trung tâm nghiên cứu Hàn Quốc"]
        },
        resources: [
        { label: "Korea.net", url: "https://www.korea.net" }
        ],
        career_prospects: "Phiên dịch viên, Giảng viên tiếng Hàn, Chuyên gia văn hóa Hàn Quốc.",
        email: "korean@hutech.edu.vn",
        phone: "028 3512 0799",
        website: "https://hutech.edu.vn"
    },

    TOURISM: {
        name: "Khoa Du lịch & Nhà hàng – Khách sạn",
        description: "Đào tạo chuyên ngành du lịch, quản lý khách sạn và dịch vụ.",
        overview: "Sinh viên được học kỹ năng quản lý, điều hành du lịch và khách sạn.",
        research: {
        labs: ["Phòng mô phỏng khách sạn", "Phòng thực hành du lịch"]
        },
        resources: [
        { label: "UNWTO", url: "https://www.unwto.org" }
        ],
        career_prospects: "Quản lý khách sạn, Điều hành tour, Chuyên viên marketing du lịch.",
        email: "tourism@hutech.edu.vn",
        phone: "028 3512 0800",
        website: "https://hutech.edu.vn"
    },

    PE: {
        name: "Khoa Giáo dục Thể chất",
        description: "Đào tạo giáo viên thể chất và nghiên cứu khoa học thể thao.",
        overview: "Sinh viên học các môn vận động, sức khỏe và giáo dục thể chất.",
        research: {
        labs: ["Phòng nghiên cứu thể thao"]
        },
        resources: [
        { label: "Science of Sport", url: "https://www.sportssci.org" }
        ],
        career_prospects: "Giáo viên thể chất, Huấn luyện viên thể thao, Chuyên gia dinh dưỡng thể thao.",
        email: "pe@hutech.edu.vn",
        phone: "028 3512 0801",
        website: "https://hutech.edu.vn"
    },

    IAS: {
        name: "Viện Khoa học Ứng dụng",
        description: "Đào tạo và nghiên cứu khoa học ứng dụng.",
        overview: "Sinh viên được tham gia các dự án nghiên cứu và ứng dụng thực tiễn.",
        research: {
        labs: ["Phòng thí nghiệm khoa học vật liệu", "Phòng thí nghiệm công nghệ sinh học"]
        },
        resources: [
        { label: "ScienceDirect", url: "https://www.sciencedirect.com" }
        ],
        career_prospects: "Kỹ sư vật liệu, Nhà nghiên cứu sinh học, Chuyên gia năng lượng tái tạo.",
        email: "ias@hutech.edu.vn",
        phone: "028 3512 0802",
        website: "https://hutech.edu.vn"
    },

    ARCH: {
        name: "Khoa Kiến trúc – Mỹ thuật",
        description: "Đào tạo kiến trúc sư, thiết kế mỹ thuật.",
        overview: "Khoa cung cấp chương trình đào tạo về kiến trúc và thiết kế nội thất.",
        research: {
        labs: ["Phòng thiết kế kiến trúc"]
        },
        resources: [
        { label: "ArchDaily", url: "https://www.archdaily.com" }
        ],
        career_prospects: "Kiến trúc sư, Nhà thiết kế nội thất, Chuyên gia quy hoạch đô thị.",
        email: "arch@hutech.edu.vn",
        phone: "028 3512 0803",
        website: "https://hutech.edu.vn"
    },

    POLITICS: {
        name: "Khoa Chính trị",
        description: "Đào tạo kiến thức chính trị, lý luận và hành chính.",
        overview: "Sinh viên được học về tư tưởng chính trị và quản lý hành chính.",
        research: {
        labs: ["Trung tâm nghiên cứu chính trị"]
        },
        resources: [
        { label: "Political Science Resources", url: "https://politicalscience.co" }
        ],
        career_prospects: "Chuyên viên chính trị, Nhà nghiên cứu, Cán bộ quản lý nhà nước.",
        email: "politics@hutech.edu.vn",
        phone: "028 3512 0804",
        website: "https://hutech.edu.vn"
    },

    ENGLISH: {
        name: "Khoa Ngôn ngữ Anh",
        description: "Đào tạo tiếng Anh và nghiên cứu văn hóa Anh – Mỹ.",
        overview: "Sinh viên được trang bị kỹ năng ngôn ngữ và giao tiếp quốc tế.",
        research: {
        labs: ["Trung tâm Anh ngữ"]
        },
        resources: [
        { label: "BBC Learning English", url: "https://www.bbc.co.uk/learningenglish" }
        ],
        career_prospects: "Giảng viên tiếng Anh, Biên dịch viên, Chuyên gia ngôn ngữ.",
        email: "english@hutech.edu.vn",
        phone: "028 3512 0805",
        website: "https://hutech.edu.vn"
    },

    CHINESE: {
        name: "Khoa Ngôn ngữ Trung Quốc",
        description: "Đào tạo tiếng Trung và nghiên cứu văn hóa Trung Quốc.",
        overview: "Khoa cung cấp chương trình đào tạo tiếng Trung và văn hóa phương Đông.",
        research: {
        labs: ["Trung tâm Hán ngữ học"]
        },
        resources: [
        { label: "Confucius Institute", url: "https://english.hanban.org" }
        ],
        career_prospects: "Giảng viên tiếng Trung, Biên dịch viên, Chuyên gia kinh tế Trung Quốc.",
        email: "chinese@hutech.edu.vn",
        phone: "028 3512 0806",
        website: "https://hutech.edu.vn"
    },

    VET: {
        name: "Khoa Thú y",
        description: "Đào tạo bác sĩ thú y và nghiên cứu y học thú y.",
        overview: "Sinh viên học về chẩn đoán, điều trị và phòng bệnh cho động vật.",
        research: {
        labs: ["Phòng thí nghiệm thú y"]
        },
        resources: [
        { label: "AVMA", url: "https://www.avma.org" }
        ],
        career_prospects: "Bác sĩ thú y, Nghiên cứu viên, Chuyên gia chăm sóc động vật.",
        email: "vet@hutech.edu.vn",
        phone: "028 3512 0807",
        website: "https://hutech.edu.vn"
    },

    ENG: {
        name: "Khoa Kỹ thuật",
        description: "Đào tạo kỹ sư đa ngành.",
        overview: "Sinh viên học về các ngành kỹ thuật điện, điện tử, cơ khí.",
        research: {
        labs: ["Phòng thí nghiệm cơ điện tử", "Phòng thí nghiệm tự động hóa"]
        },
        resources: [
        { label: "IEEE Xplore", url: "https://ieeexplore.ieee.org" }
        ],
        career_prospects: "Kỹ sư điện, Kỹ sư cơ khí, Kỹ sư tự động hóa.",
        email: "eng@hutech.edu.vn",
        phone: "028 3512 0808",
        website: "https://hutech.edu.vn"
    },

    CIVIL: {
        name: "Khoa Xây dựng",
        description: "Đào tạo kỹ sư xây dựng và kiến trúc sư.",
        overview: "Sinh viên học các môn liên quan đến thiết kế và thi công xây dựng.",
        research: {
        labs: ["Phòng thí nghiệm xây dựng"]
        },
        resources: [
        { label: "Civil Engineering Portal", url: "https://civilengineeringportal.com" }
        ],
        career_prospects: "Kỹ sư xây dựng, Kiến trúc sư, Quản lý dự án xây dựng.",
        email: "civil@hutech.edu.vn",
        phone: "028 3512 0809",
        website: "https://hutech.edu.vn"
    },

    MKTINT: {
        name: "Khoa Marketing Quốc tế",
        description: "Đào tạo chuyên ngành marketing, thương mại quốc tế.",
        overview: "Sinh viên học về quản trị marketing, thương mại và kinh doanh quốc tế.",
        research: {
        labs: ["Trung tâm nghiên cứu marketing"]
        },
        resources: [
        { label: "American Marketing Association", url: "https://www.ama.org" }
        ],
        career_prospects: "Chuyên viên marketing, Quản lý thương hiệu, Chuyên gia thương mại quốc tế.",
        email: "mktint@hutech.edu.vn",
        phone: "028 3512 0810",
        website: "https://hutech.edu.vn"
    }
    };
