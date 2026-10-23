-- 003: the AI4I 2020 dataset as a table, for the copilot's data tools (process_data.py).
--
-- One row per recorded machining process. The quantities the failure rules
-- reason with are generated columns: Postgres computes them from the sensor
-- columns on insert, so they can't disagree with them.

CREATE TABLE ai4i_processes (
    udi                     integer PRIMARY KEY,
    product_id              text NOT NULL,
    product_type            char(1) NOT NULL CHECK (product_type IN ('L', 'M', 'H')),
    air_temperature_k       real NOT NULL,
    process_temperature_k   real NOT NULL,
    rotational_speed_rpm    integer NOT NULL,
    torque_nm               real NOT NULL,
    tool_wear_min           integer NOT NULL,
    machine_failure         boolean NOT NULL,
    twf                     boolean NOT NULL,
    hdf                     boolean NOT NULL,
    pwf                     boolean NOT NULL,
    osf                     boolean NOT NULL,
    rnf                     boolean NOT NULL,

    temperature_difference_k real GENERATED ALWAYS AS (process_temperature_k - air_temperature_k) STORED,
    power_w                  real GENERATED ALWAYS AS (torque_nm * rotational_speed_rpm * 2 * pi() / 60) STORED,
    strain_min_nm            real GENERATED ALWAYS AS (tool_wear_min * torque_nm) STORED,

    -- The quantities the failure rules reason with - temperature difference,
    -- speed, power, wear x torque and tool wear - each scaled to mean 0 and
    -- standard deviation 1 over the whole table (set by process_data.load).
    -- Scaled, so that a kelvin and a watt weigh alike: unscaled, power (~6,000)
    -- would decide every distance on its own. These rather than the raw
    -- readings, because neighbours here share a failure mode more often
    -- (evals/failure_model_eval.py).
    features                vector(5)
);

CREATE INDEX ai4i_processes_type_idx ON ai4i_processes (product_type);

-- The scaling, for turning a query's readings into the same space.
CREATE VIEW ai4i_scaling AS
SELECT
    avg(temperature_difference_k)     AS diff_mean,    stddev_pop(temperature_difference_k) AS diff_sd,
    avg(rotational_speed_rpm)         AS rpm_mean,     stddev_pop(rotational_speed_rpm)     AS rpm_sd,
    avg(power_w)                      AS power_mean,   stddev_pop(power_w)                  AS power_sd,
    avg(strain_min_nm)                AS strain_mean,  stddev_pop(strain_min_nm)            AS strain_sd,
    avg(tool_wear_min)                AS wear_mean,    stddev_pop(tool_wear_min)            AS wear_sd
FROM ai4i_processes;
