def none_or_float(value):
    if value == 'None':
        return None
    elif value is None:
        return None
    else:
        return float(value)
