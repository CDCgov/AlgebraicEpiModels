# The school calendar that switches a school contact layer on model time.

"""
    SchoolCalendar(first_date, values)

Daily fraction of school contacts active, from `first_date`, each in `[0, 1]`.
"""
struct SchoolCalendar
    first_date::Date
    values::Vector{Float64}
    function SchoolCalendar(first_date::Date, values::AbstractVector)
        isempty(values) && throw(ArgumentError("a school calendar needs at least one day"))
        all(v -> isfinite(v) && 0 <= v <= 1, values) ||
            throw(ArgumentError("school-open fractions must lie in [0, 1]"))
        return new(first_date, Float64.(values))
    end
end

last_date(cal::SchoolCalendar) = cal.first_date + Day(length(cal.values) - 1)

"""
    load_school_calendar(path) -> SchoolCalendar

Read `date,school_open_fraction` rows, which must be consecutive days.
"""
function load_school_calendar(path::AbstractString)
    rows = _read_table(path, "date,school_open_fraction", "school calendar")
    dates = [Date(date) for (date, _) in rows]
    all(==(Day(1)), diff(dates)) || error("`$path` dates are not consecutive days")
    return SchoolCalendar(first(dates), [parse(Float64, value) for (_, value) in rows])
end

"""
    AnchoredCalendar(cal, start_date)

The calendar on model time, `t` days after `start_date`, piecewise constant per day and held at
its first and last values outside its range.
"""
struct AnchoredCalendar
    values::Vector{Float64}
    offset::Int
end
AnchoredCalendar(cal::SchoolCalendar, start_date::Date) =
    AnchoredCalendar(cal.values, Dates.value(start_date - cal.first_date))

@inline (cal::AnchoredCalendar)(t) = cal.values[clamp(floor(Int, t) + cal.offset + 1, 1, length(cal.values))]

"""
    assert_calendar_covers(cal, from::Date, to::Date)

Error unless the calendar has a value for every day in `[from, to]`.
"""
function assert_calendar_covers(cal::SchoolCalendar, from::Date, to::Date)
    cal.first_date <= from && last_date(cal) >= to ||
        error("school calendar covers $(cal.first_date) to $(last_date(cal)); the model needs $from to $to")
    return nothing
end
