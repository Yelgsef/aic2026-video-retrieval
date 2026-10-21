from aic_pipeline.task_router import route_query, split_events


def test_parenthesized_semicolon_trake_events() -> None:
    query = (
        "(1) Hai MC giới thiệu bản tin; "
        "(2) Một giếng khoan phun nước và nhiều người quan sát; "
        "(3) Các bác sĩ thực hiện ca ghép tim."
    )
    events = split_events(query)
    assert len(events) == 3
    assert events[0] == "Hai MC giới thiệu bản tin"
    assert events[1].startswith("Một giếng khoan")
    assert events[2].startswith("Các bác sĩ")
    assert route_query(query).task == "TRAKE"


def test_e_markers_and_parentheses_inside_event() -> None:
    query = "E1: Người đàn ông nâng hộp (màu đỏ); E2: Ông ấy mở hộp; E3: Ông lấy quà ra"
    events = split_events(query)
    assert len(events) == 3
    assert "(màu đỏ)" in events[0]


def test_multiline_e_markers_without_colons_are_trake() -> None:
    query = (
        "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.\n"
        "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ.\n"
        "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân."
    )
    plan = route_query(query)
    assert plan.task == "TRAKE"
    assert len(plan.events) == 3
    assert "hai con rồng" in plan.events[0]


def test_single_ai_word_is_a_question_but_hai_is_not() -> None:
    assert route_query("Ai đang cầm dùi?").task == "QA"
    assert route_query("E1: Hai con rồng xuất hiện; E2: Con lân xoay vòng").task == "TRAKE"
