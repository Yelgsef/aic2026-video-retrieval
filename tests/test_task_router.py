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
