from optical_rectification import observable

def test_observable():
    try:
        val = observable.conversion_efficiency()
        assert isinstance(val, float), "conversion efficiency should be a number"
    except AssertionError as a:
        print(f"AssertionError: ", {a})
    return val

print(test_observable())
